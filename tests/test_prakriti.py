"""Prakriti scoring — PROVISIONAL content, tested as software.

These test the *mechanics*: that the items are asked, that a count produces a
constitution, that too few answers produce none. They deliberately do not assert that
any particular answer set indicates any particular prakriti, because that is the
clinical judgement this module is standing in for until a practitioner replaces it.
"""
import json
from pathlib import Path

from sanjeevani_ml.ayush.prakriti import DOSHAS, PRAKRITI_ITEMS, derive, score
from sanjeevani_ml.dialogue.engine import DialogueEngine
from sanjeevani_ml.schemas import Answer
from sanjeevani_ml.service.main import CONDITION_MAP_PATH, ONTOLOGY_PATH


def test_the_provisional_status_is_recorded_in_the_ontology():
    """The one thing that must not get lost: this content is unreviewed.

    If someone deletes the marker, they have to do it deliberately.
    """
    ontology = json.loads(Path(ONTOLOGY_PATH).read_text(encoding="utf-8"))
    section = next(s for s in ontology["sections"] if s["id"] == "prakriti_assessment")
    assert "PROVISIONAL" in section["$validation"]
    assert "NOT PRACTITIONER REVIEWED" in section["$validation"]
    assert "PROVISIONAL" in ontology["$review_status"]["prakriti_assessment"]


def test_every_scored_item_exists_in_the_ontology():
    """A scored item the interview never asks silently shrinks the sample."""
    ontology = json.loads(Path(ONTOLOGY_PATH).read_text(encoding="utf-8"))
    section = next(s for s in ontology["sections"] if s["id"] == "prakriti_assessment")
    asked = {q["id"] for q in section["questions"]}
    assert set(PRAKRITI_ITEMS) == asked


def test_every_item_offers_exactly_the_three_doshas():
    ontology = json.loads(Path(ONTOLOGY_PATH).read_text(encoding="utf-8"))
    section = next(s for s in ontology["sections"] if s["id"] == "prakriti_assessment")
    for question in section["questions"]:
        options = [o["id"] for o in question["options"]]
        assert options == list(DOSHAS), f"{question['id']} offers {options}"
        for option in question["options"]:
            assert set(option["label"]) >= {"en", "hi"}, f"{question['id']} missing a language"


def test_a_clear_constitution_is_reported_singly():
    answers = {item: "vata" for item in PRAKRITI_ITEMS}
    result = derive(answers)
    assert result["dominant"] == "vata"
    assert result["dual"] is False


def test_an_even_split_is_reported_as_dual():
    """Dual constitutions are common. Forcing a single answer would be a false
    precision the arithmetic cannot support."""
    answers = dict.fromkeys(PRAKRITI_ITEMS[:4], "pitta")
    answers.update(dict.fromkeys(PRAKRITI_ITEMS[4:], "kapha"))
    result = derive(answers)
    assert result["dual"] is True
    assert "-" in result["dominant"]


def test_too_few_answers_yields_nothing():
    """A constitution read off two questions is not a constitution."""
    assert derive({"pa.build": "vata"}) is None
    assert derive({}) is None


def test_the_result_carries_its_own_warning():
    """Whatever consumes this must be able to see it is unreviewed without
    knowing where it came from."""
    result = derive({item: "kapha" for item in PRAKRITI_ITEMS})
    assert result["provisional"] is True
    assert "PROVISIONAL" in result["validation"]
    assert "not a diagnosis" in result["validation"].lower()


def test_unrecognised_answers_are_ignored_not_counted():
    answers = {item: "not_a_dosha" for item in PRAKRITI_ITEMS}
    assert score(answers) == {"vata": 0, "pitta": 0, "kapha": 0}


def test_an_ayush_interview_asks_every_prakriti_item():
    """Scoring is useless if the interview never collects the answers."""
    engine = DialogueEngine(str(ONTOLOGY_PATH), condition_map_path=str(CONDITION_MAP_PATH))
    question = engine.start_interview(session_id="p", language="en", mode="ayush")
    asked, guard = [], 0
    while question is not None and guard < 80:
        asked.append(question.question_id)
        question = engine.submit_answer(
            "p", Answer(question_id=question.question_id, value="vata")
        )
        guard += 1
    assert set(PRAKRITI_ITEMS) <= set(asked), (
        f"missing: {sorted(set(PRAKRITI_ITEMS) - set(asked))}"
    )


def test_an_allopathic_interview_asks_none_of_them():
    """Constitution is an AYUSH assessment. An allopathic patient is not asked it."""
    engine = DialogueEngine(str(ONTOLOGY_PATH), condition_map_path=str(CONDITION_MAP_PATH))
    question = engine.start_interview(session_id="a", language="en", mode="allopathic")
    asked, guard = [], 0
    while question is not None and guard < 80:
        asked.append(question.question_id)
        question = engine.submit_answer(
            "a", Answer(question_id=question.question_id, value="no")
        )
        guard += 1
    assert not [q for q in asked if q.startswith("pa.")]
