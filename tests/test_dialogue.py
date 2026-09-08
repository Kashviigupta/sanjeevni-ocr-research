"""Dialogue engine tests — driven against the REAL ontology, not a fixture.

The engine decides what a patient is asked. It is the most load-bearing logic in
the repo, and it shipped without tests; both bugs these cover were invisible until
something actually walked the real file:

  1. `load_ontology` accepted only `questions`, but `dashavidha_pariksha` keys its
     items under `parameters`. The engine could not even be CONSTRUCTED — in either
     mode, because validation runs before mode filtering.
  2. A section whose only condition was `mode: "ayush"` passed the mode check and
     then fell through, having neither `always` nor `trigger`. An AYUSH interview
     silently ran the allopathic question set: the differentiator, quietly absent.

Hence the rule these encode — **always drive the real ontology.** A fixture ontology
would have passed while both bugs sat in main.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sanjeevani_ml.dialogue.engine import DialogueEngine
from sanjeevani_ml.dialogue.ontology_loader import (
    derived_item_ids,
    flatten_questions,
    load_ontology,
    section_items,
)
from sanjeevani_ml.schemas import Answer

ONTOLOGY = Path(__file__).resolve().parents[2] / "shared" / "clinical" / "history-ontology.json"

#: Answers that carry an interview past the always-on sections without tripping a
#: red flag, so a test can isolate whatever it is actually about.
_QUIET = {
    "rf.stroke": ["none"],
    "px.surgery": "no",
    "rx.current": "no",
    "rx.allergy": "no",
    "*": "x",
}


def walk(mode: str, script: dict, limit: int = 80):
    """Run an interview to completion. Returns (question_ids_asked, transcript)."""
    engine = DialogueEngine(str(ONTOLOGY))
    session = f"test-{mode}"
    question = engine.start_interview(session_id=session, language="hi", mode=mode)

    asked: list[str] = []
    for _ in range(limit):
        if question is None:
            break
        asked.append(question.question_id)
        value = script.get(question.question_id, script.get("*"))
        question = engine.submit_answer(
            session, Answer(question_id=question.question_id, value=value, via_touch=True)
        )
    else:
        pytest.fail(f"interview did not terminate within {limit} questions")

    return asked, engine.get_transcript(session)


class TestLoadsTheRealOntology:
    def test_engine_constructs_against_the_real_ontology(self):
        """The regression guard. This is the bug that made the engine unusable."""
        assert DialogueEngine(str(ONTOLOGY)) is not None

    def test_sections_keyed_by_parameters_are_read(self):
        """`dashavidha_pariksha` uses `parameters`, every other section `questions`."""
        ontology = load_ontology(ONTOLOGY)
        dashavidha = next(s for s in ontology["sections"] if s["id"] == "dashavidha_pariksha")
        assert "parameters" in dashavidha, "the ontology changed shape — update the loader"
        assert section_items(dashavidha), "parameters were not read as askable items"

    def test_derived_items_are_never_askable(self):
        """Prakriti is scored, Vikriti computed, Vaya from DOB — none can be asked."""
        ontology = load_ontology(ONTOLOGY)
        derived = set(derived_item_ids(ontology))
        assert {"dp.prakriti", "dp.vikriti", "dp.vaya"} <= derived
        assert derived.isdisjoint(flatten_questions(ontology)), \
            "a derived item reached the askable set — the kiosk would ask a patient their Vikriti"


class TestAdaptiveBranching:
    def test_chest_pain_pulls_in_the_socrates_probe(self):
        asked, _ = walk("allopathic", {**_QUIET, "cc.area": "chest", "cc.primary": "pain"})
        assert any(q.startswith("soc.") for q in asked)

    def test_a_non_pain_complaint_skips_socrates(self):
        """Adaptive means skipping too — asking a rash about radiation wastes the queue."""
        asked, _ = walk("allopathic", {**_QUIET, "cc.area": "skin", "cc.primary": "skin"})
        assert not any(q.startswith("soc.") for q in asked)

    def test_follow_up_fires_on_yes_and_not_on_no(self):
        asked_yes, _ = walk("allopathic", {**_QUIET, "cc.primary": "checkup", "px.surgery": "yes"})
        asked_no, _ = walk("allopathic", {**_QUIET, "cc.primary": "checkup", "px.surgery": "no"})
        assert "px.surgery_detail" in asked_yes
        assert "px.surgery_detail" not in asked_no


class TestAyushMode:
    """The differentiator. Ministry of Ayush and AIIA are judging this."""

    def test_ayush_mode_asks_dashavidha_pariksha(self):
        asked, _ = walk("ayush", {**_QUIET, "cc.area": "chest", "cc.primary": "pain"})
        parameters = [q for q in asked if q.startswith("dp.")]
        assert parameters, "ayush mode ran the allopathic question set"
        # 10 declared, 3 derived — the rest must actually be put to the patient.
        assert len(parameters) == 7, f"expected 7 askable parameters, got {parameters}"

    def test_ayush_mode_asks_ahara_vihara(self):
        asked, _ = walk("ayush", {**_QUIET, "cc.primary": "checkup"})
        assert any(q.startswith("av.") for q in asked)

    def test_allopathic_mode_excludes_ayush_sections(self):
        asked, _ = walk("allopathic", {**_QUIET, "cc.area": "chest", "cc.primary": "pain"})
        assert not any(q.startswith("dp.") or q.startswith("av.") for q in asked)


class TestRedFlagsDoNotInterruptTheInterview:
    """Non-negotiable: a red flag alerts triage and the interview carries on."""

    def test_a_fired_red_flag_does_not_end_the_interview(self):
        asked, transcript = walk("allopathic", {
            **_QUIET, "cc.area": "chest", "cc.primary": "pain",
            "soc.associations": ["breathless"],
        })
        assert transcript.red_flags, "chest pain with breathlessness should raise a flag"
        assert transcript.completed, "the interview must run to completion anyway"
        flagged_at = asked.index("soc.associations")
        assert flagged_at < len(asked) - 1, "questions must continue after the flag"

    def test_red_flag_explanations_stay_physician_facing(self):
        _, transcript = walk("allopathic", {
            **_QUIET, "cc.area": "chest", "cc.primary": "pain",
            "soc.associations": ["breathless"],
        })
        for flag in transcript.red_flags:
            assert flag.reason, "a flag without a reason cannot be routed"
            # The kiosk renders `Question`, never a RedFlag — this asserts the shape
            # keeps explanation separate from anything patient-facing.
            assert not hasattr(flag, "patient_message")
