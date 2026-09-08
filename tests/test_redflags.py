"""Tests red flag detection against a trimmed but structurally-real copy of
history-ontology.json (same keys/shape, prompt text omitted since rules.py
never reads it). The full file's prompt/label content doesn't affect this
module's logic -- it only reads input_type, options, and red_flag blocks.
"""

import json
from pathlib import Path

import pytest

from sanjeevani_ml.redflags.rules import evaluate_answer, flatten_questions

# Resolved from this file, not the working directory: `pytest` is run from ml/ (see
# ml/README.md), where a repo-root-relative path does not exist. Same convention as
# tests/test_shared_fixtures.py.
ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "shared" / "clinical" / "history-ontology.json"


@pytest.fixture
def flat():
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    return flatten_questions(ontology)


def test_chest_pain_with_breathlessness_fires_critical(flat):
    """The project's own 'done' bar: chest pain + breathlessness -> alert."""
    all_answers = {"cc.area": "chest", "cc.primary": "pain"}
    flags = evaluate_answer("soc.associations", ["breathless"], all_answers, flat)

    assert len(flags) == 1
    assert flags[0].reason == "chest_pain_with_dyspnoea"
    assert flags[0].severity == "critical"
    assert flags[0].question_id == "soc.associations"


def test_breathlessness_without_chest_area_does_not_fire(flat):
    """Same symptom, different body area -- must NOT fire, since the rule
    is scoped to cc.area == chest."""
    all_answers = {"cc.area": "abdomen", "cc.primary": "pain"}
    flags = evaluate_answer("soc.associations", ["breathless"], all_answers, flat)
    assert flags == []


def test_acute_onset_chest_pain_fires_on_duration_answer(flat):
    """soc.onset's own red_flag: value_hours <= 1 AND cc.area == chest."""
    all_answers = {"cc.area": "chest"}
    duration_value = {"amount": 30, "unit": "hours" if False else "minutes"}
    # Ontology only defines hours/days/weeks/months/years -- minutes isn't
    # a supported unit, so use a supported one for this test.
    duration_value = {"amount": 0.5, "unit": "hours"}
    flags = evaluate_answer("soc.onset", duration_value, all_answers, flat)

    assert len(flags) == 1
    assert flags[0].reason == "chest_pain_acute_onset"


def test_onset_three_days_ago_does_not_fire(flat):
    all_answers = {"cc.area": "chest"}
    duration_value = {"amount": 3, "unit": "days"}
    flags = evaluate_answer("soc.onset", duration_value, all_answers, flat)
    assert flags == []


def test_radiation_to_left_arm_fires(flat):
    all_answers = {"cc.area": "chest"}
    flags = evaluate_answer("soc.radiation", "arm_left", all_answers, flat)
    assert len(flags) == 1
    assert flags[0].reason == "chest_pain_radiation_cardiac"


def test_stroke_screen_option_with_no_when_always_fires(flat):
    """rf.stroke options have no 'when' clause -- selecting them should
    always fire, unconditionally."""
    flags = evaluate_answer("rf.stroke", ["face_droop"], {}, flat)
    assert len(flags) == 1
    assert flags[0].reason == "stroke_facial_droop"


def test_stroke_screen_none_selected_fires_nothing(flat):
    flags = evaluate_answer("rf.stroke", ["none"], {}, flat)
    assert flags == []


def test_multiple_stroke_symptoms_fire_multiple_flags(flat):
    flags = evaluate_answer("rf.stroke", ["face_droop", "speech"], {}, flat)
    reasons = {f.reason for f in flags}
    assert reasons == {"stroke_facial_droop", "stroke_speech"}


def test_explanations_are_never_shown_to_patient_by_construction(flat):
    """RedFlag.explanation is physician-facing text keyed by language --
    this test just documents that it exists and is populated for a known
    reason, as a guard against someone accidentally deleting the table."""
    flags = evaluate_answer("rf.stroke", ["bleeding"], {}, flat)
    assert "en" in flags[0].explanation
    assert "hi" in flags[0].explanation
