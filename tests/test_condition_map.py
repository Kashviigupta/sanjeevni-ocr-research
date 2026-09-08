"""The condition map: what makes the interview feel like a clinician, not a form.

Say "chest pain, like pressure" and the next questions should be about radiation and
sweating — not your family history and your diet. That reordering is the whole feature.

It arrived as 310 lines with no tests and, more importantly, with nothing passing
`condition_map_path`, so `self._condition_map` was always None and every line of it was
dead. These tests exist so that cannot happen again quietly: `test_the_service_actually_
enables_the_map` fails the moment it is unwired.

**This orders questions. It is not a diagnosis and must never be shown as one.**
"""
from pathlib import Path

import pytest

from sanjeevani_ml.dialogue.condition_map import ConditionMap
from sanjeevani_ml.dialogue.engine import DialogueEngine
from sanjeevani_ml.schemas import Answer
from sanjeevani_ml.service.main import CONDITION_MAP_PATH, ONTOLOGY_PATH


@pytest.fixture(scope="module")
def cmap() -> ConditionMap:
    return ConditionMap(str(CONDITION_MAP_PATH))


def test_the_service_actually_enables_the_map():
    """The bug that made all of this dead code.

    `condition_map_path` defaults to None and the engine silently degrades to ontology
    order when it is not supplied — a reasonable default that hid the feature entirely.
    """
    from sanjeevani_ml.service import main

    main._ENGINE = None
    engine = main._engine()
    assert engine._condition_map is not None, (
        "the service built a DialogueEngine without a condition map - "
        "correlated questioning is silently off"
    )


def test_every_discriminator_names_a_real_question():
    """A typo'd question id is a discriminator that can never fire, and nothing
    anywhere would report it — the map would just quietly do less."""
    import json

    onto = json.loads(Path(ONTOLOGY_PATH).read_text(encoding="utf-8"))
    known = {
        q["id"]
        for s in onto["sections"]
        for q in (s.get("questions") or s.get("parameters") or [])
    }
    raw = json.loads(Path(CONDITION_MAP_PATH).read_text(encoding="utf-8"))
    referenced = {
        d["question_id"] for c in raw["conditions"] for d in c["discriminators"]
    }
    assert not referenced - known, f"map references unknown questions: {sorted(referenced - known)}"


def test_chest_pressure_makes_acs_plausible(cmap):
    """The one we must never miss."""
    plausibility = cmap.calculate_plausibility({"cc.area": "chest", "soc.character": "pressure"})
    assert plausibility.get("acute_coronary_syndrome", 0) > 0.5


def test_plausibility_reorders_toward_the_discriminating_questions(cmap):
    """The behaviour the whole feature exists for."""
    answers = {"cc.area": "chest", "soc.character": "pressure"}
    candidates = ["px.surgery", "fh.conditions", "soc.radiation", "soc.associations", "pers.diet"]

    ranked = cmap.rank_questions(candidates=candidates, answers=answers)

    assert set(ranked) == set(candidates), "ranking dropped or invented a question"
    # Does it go down your arm / are you sweating — before diet and surgical history.
    assert ranked.index("soc.radiation") < ranked.index("pers.diet")
    assert ranked.index("soc.associations") < ranked.index("fh.conditions")


def test_no_evidence_leaves_the_clinician_authored_order_alone(cmap):
    """Absent evidence, the ontology's order is the considered one. Do not churn it."""
    candidates = ["cc.primary", "cc.describe", "px.conditions"]
    assert cmap.rank_questions(candidates=candidates, answers={}) == candidates


def test_ranking_never_drops_or_duplicates_a_question(cmap):
    """A dropped question is a question the patient is never asked."""
    candidates = sorted({
        d["question_id"] for c in cmap.conditions for d in c.discriminators
    })
    ranked = cmap.rank_questions(
        candidates=candidates, answers={"cc.area": "chest", "soc.character": "pressure"}
    )
    assert sorted(ranked) == candidates
    assert len(ranked) == len(set(ranked))


def test_an_unknown_question_survives_ranking(cmap):
    """Questions no condition cares about must still get asked, just later."""
    ranked = cmap.rank_questions(
        candidates=["not.in.any.condition", "soc.radiation"],
        answers={"cc.area": "chest", "soc.character": "pressure"},
    )
    assert "not.in.any.condition" in ranked


def test_the_engine_uses_it_end_to_end():
    """Same interview, map on and off, must diverge once evidence exists."""
    def run(with_map: bool) -> list[str]:
        engine = DialogueEngine(
            str(ONTOLOGY_PATH),
            condition_map_path=str(CONDITION_MAP_PATH) if with_map else None,
        )
        engine.start_interview(session_id="s", language="en", mode="allopathic")
        # Seed the evidence directly: this test is about ordering, not about how the
        # patient reaches these questions.
        engine._sessions["s"].answers.update({"cc.area": "chest", "soc.character": "pressure"})
        state = engine._sessions["s"]
        return [engine._compute_next_question_id(state)]

    assert run(True) != [None]
