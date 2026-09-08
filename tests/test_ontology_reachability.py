"""Every declared red flag must be reachable by an actual patient.

Two critical flags were declared, tested at the unit level, and could never fire in
a real interview:

* `chest_pain_radiation_cardiac` fired on `value in [arm_left, jaw, back]`, but
  `soc.radiation` declared no regions, so no client could offer those options.
* `chest_pain_acute_onset` reads `value_hours`, but `Answer.value` could not carry a
  duration at all - the schema rejected `{"amount": 0.5, "unit": "hours"}`.

Both passed every existing test, because the rule evaluator was tested with
hand-built inputs rather than answers a patient could actually produce. These tests
walk the real engine instead.
"""
import json
from pathlib import Path

import pytest

from sanjeevani_ml.dialogue.engine import DialogueEngine
from sanjeevani_ml.schemas import Answer
from sanjeevani_ml.service.main import CONDITION_MAP_PATH, ONTOLOGY_PATH

BRANCHES = {
    "pain": "socrates", "fever": "fever", "cough_breath": "respiratory",
    "digestive": "digestive", "weakness": "constitutional", "skin": "dermatology",
}


def _ontology():
    return json.loads(Path(ONTOLOGY_PATH).read_text(encoding="utf-8"))


def _items(section):
    return section.get("questions") or section.get("parameters") or []


def _interview(complaint, answers):
    engine = DialogueEngine(str(ONTOLOGY_PATH), condition_map_path=str(CONDITION_MAP_PATH))
    question = engine.start_interview(session_id="t", language="en", mode="allopathic")
    asked, guard = [], 0
    while question is not None and guard < 80:
        asked.append(question.question_id)
        value = answers.get(question.question_id,
                            complaint if question.question_id == "cc.primary" else "no")
        question = engine.submit_answer("t", Answer(question_id=question.question_id, value=value))
        guard += 1
    return asked, engine.get_transcript("t")


def test_every_complaint_branch_resolves_to_a_real_section():
    """A branch naming a section that does not exist silently drops the patient onto
    the generic spine - they are simply never asked about their actual complaint."""
    o = _ontology()
    sections = {s["id"] for s in o["sections"]}
    question_ids = {q["id"] for s in o["sections"] for q in _items(s)}
    missing = []
    for section in o["sections"]:
        for q in _items(section):
            if q["id"] != "cc.primary":
                continue
            for option in q.get("options", []):
                target = option.get("branch")
                if target and target not in sections and target not in question_ids:
                    missing.append(f"{option['id']} -> {target}")
    assert not missing, f"cc.primary branches to nothing: {missing}"


@pytest.mark.parametrize("complaint", sorted(BRANCHES))
def test_each_complaint_gets_its_own_questions(complaint):
    """Fever is the commonest OPD complaint in India. It once drew sixteen questions
    and not one of them was about the fever."""
    asked, _ = _interview(complaint, {})
    o = _ontology()
    section = next(s for s in o["sections"] if s["id"] == BRANCHES[complaint])
    branch_ids = {q["id"] for q in _items(section)}
    assert branch_ids & set(asked), f"{complaint} asked nothing from {BRANCHES[complaint]}"


def test_duration_answers_survive_the_schema():
    """The type that made two time-based red flags unreachable."""
    answer = Answer(question_id="soc.onset", value={"amount": 0.5, "unit": "hours"})
    assert answer.value.amount == 0.5
    assert answer.value.unit == "hours"

    with pytest.raises(ValueError):
        Answer(question_id="soc.onset", value={"amount": 1, "unit": "fortnights"})


def test_every_duration_question_declares_its_units():
    """Without units the client invents them, and `value_hours` compares against the
    wrong threshold - or never fires."""
    for section in _ontology()["sections"]:
        for q in _items(section):
            if q.get("input_type") == "duration":
                assert q.get("units"), f"{q['id']} is a duration with no units"


def test_every_body_map_declares_its_regions():
    """A body map with no regions renders an empty screen the patient cannot answer."""
    for section in _ontology()["sections"]:
        for q in _items(section):
            if q.get("input_type") == "body_map":
                assert q.get("regions"), f"{q['id']} is a body_map with no regions"


def test_a_red_flag_can_only_fire_on_an_option_that_exists():
    """The bug this file is named for: a rule matching values no client can send."""
    problems = []
    for section in _ontology()["sections"]:
        for q in _items(section):
            flag = q.get("red_flag") or {}
            when = flag.get("when", "")
            if " in [" not in when:
                continue
            wanted = {v.strip() for v in when.split(" in [", 1)[1].split("]")[0].split(",")}
            offerable = set(q.get("regions") or []) | {o["id"] for o in q.get("options", [])}
            unreachable = wanted - offerable
            if unreachable:
                problems.append(f"{q['id']}: rule matches {sorted(unreachable)}, "
                                f"but the patient can only choose {sorted(offerable)}")
    assert not problems, "; ".join(problems)


CRITICAL_PATHS = [
    ("pain", {"cc.area": "chest", "soc.radiation": "arm_left"}, "chest_pain_radiation_cardiac"),
    ("pain", {"cc.area": "chest", "soc.onset": {"amount": 0.5, "unit": "hours"}},
     "chest_pain_acute_onset"),
    ("fever", {"fev.associated": ["neck_stiffness"]}, "fever_with_neck_stiffness"),
    ("fever", {"fev.onset": {"amount": 10, "unit": "days"}}, "prolonged_fever"),
    ("cough_breath", {"resp.sputum": "blood"}, "haemoptysis"),
    ("cough_breath", {"resp.breathless": "at_rest"}, "dyspnoea_at_rest"),
    ("digestive", {"dig.blood": "in_vomit"}, "gi_bleeding"),
    ("digestive", {"dig.urine": "very_little"}, "dehydration_reduced_urine"),
    ("digestive", {"dig.associated": ["yellow_eyes"]}, "jaundice"),
    ("weakness", {"const.bleeding": ["gums"]}, "haemorrhage_unexplained"),
    ("skin", {"derm.associated": ["numb_patch"]}, "skin_numb_patch"),
    ("skin", {"derm.spread": "spreading", "derm.associated": ["fever"]},
     "skin_spreading_with_fever"),
]


@pytest.mark.parametrize("complaint,answers,expected", CRITICAL_PATHS)
def test_the_flag_fires_for_a_patient_who_would_raise_it(complaint, answers, expected):
    """Walks the real engine, not the rule evaluator in isolation. A flag that only
    fires in a unit test is not a flag."""
    _, transcript = _interview(complaint, answers)
    raised = {f.reason for f in transcript.red_flags}
    assert expected in raised, f"{expected} did not fire; got {sorted(raised)}"
