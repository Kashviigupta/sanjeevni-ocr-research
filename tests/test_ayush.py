from __future__ import annotations

import pytest

from sanjeevani_ml.ayush.dashavidha import (
    ADDITIONAL_AYUSH_ORDER,
    DASHAVIDHA_ORDER,
    parameter_question_map,
)
from sanjeevani_ml.ayush.validator import AYUSHValidator
from sanjeevani_ml.dialogue.condition_map import ConditionMap


# ============================================================
# Helpers
# ============================================================

#: The real ids the dialogue engine emits, read from the ontology rather than invented.
#: An earlier hand-written map used an `ayush.dashavidha.*` namespace that appears
#: nowhere in the ontology, so the validator was exercised entirely against ids no
#: interview could ever produce.
DASHAVIDHA_QUESTION_MAP = parameter_question_map()


def make_answers():
    """
    Creates placeholder answers.

    These are intentionally NOT Ayurvedic clinical values.
    They are only used to test the software flow.
    """
    return {
        question_id: "placeholder"
        for question_id in DASHAVIDHA_QUESTION_MAP.values()
    }


# ============================================================
# 1. DASHAVIDHA STRUCTURE
# ============================================================

def test_dashavidha_contains_ten_parameters():
    assert len(DASHAVIDHA_ORDER) == 10


def test_dashavidha_parameter_order():
    assert DASHAVIDHA_ORDER == (
        "prakriti",
        "vikriti",
        "sara",
        "samhanana",
        "pramana",
        "satmya",
        "sattva",
        "ahara_shakti",
        "vyayama_shakti",
        "vaya",
    )


def test_additional_ayush_parameters_match_the_ontology():
    """Derived, not listed.

    The hand-written version named `agni` and `samprapti`, which the ontology does not
    ask about at all - so `is_complete()` could never return True however thoroughly a
    patient answered - and it missed `diet` and `sleep`, which it does ask.
    """
    assert ADDITIONAL_AYUSH_ORDER == ("koshtha", "diet", "sleep", "nidana")


def test_every_parameter_maps_to_a_question_that_exists():
    """A validator built against invented ids validates nothing that happens in an
    interview."""
    import json
    from pathlib import Path

    from sanjeevani_ml.ayush.dashavidha import _find_ontology

    ontology = json.loads(Path(_find_ontology()).read_text(encoding="utf-8"))
    real = {
        item["id"]
        for section in ontology["sections"]
        for item in (section.get("parameters") or section.get("questions") or [])
    }
    for parameter, question_id in parameter_question_map().items():
        assert question_id in real, f"{parameter} maps to {question_id}, which does not exist"


# ============================================================
# 2. AYUSH VALIDATOR
# ============================================================

def test_empty_ayush_interview_is_incomplete():
    validator = AYUSHValidator({})

    missing = validator.missing_dashavidha(
        DASHAVIDHA_QUESTION_MAP
    )

    assert len(missing) == 10


def test_answered_parameter_is_removed_from_missing():
    answers = {
        DASHAVIDHA_QUESTION_MAP["prakriti"]: "placeholder"
    }

    validator = AYUSHValidator(answers)

    missing = validator.missing_dashavidha(
        DASHAVIDHA_QUESTION_MAP
    )

    assert "prakriti" not in missing
    assert len(missing) == 9


def test_all_dashavidha_parameters_are_answered():
    answers = {
        question_id: "placeholder"
        for parameter, question_id in DASHAVIDHA_QUESTION_MAP.items()
        if parameter in DASHAVIDHA_ORDER
    }

    validator = AYUSHValidator(answers)

    missing = validator.missing_dashavidha(
        DASHAVIDHA_QUESTION_MAP
    )

    assert missing == []


def test_additional_ayush_parameters_are_detected():
    answers = {
        DASHAVIDHA_QUESTION_MAP["koshtha"]: "placeholder",
        DASHAVIDHA_QUESTION_MAP["diet"]: "placeholder",
    }

    validator = AYUSHValidator(answers)

    missing = validator.missing_additional(
        DASHAVIDHA_QUESTION_MAP
    )

    assert "koshtha" not in missing
    assert "diet" not in missing
    assert "sleep" in missing
    assert "nidana" in missing


def test_complete_ayush_interview():
    answers = make_answers()

    validator = AYUSHValidator(answers)

    assert validator.is_complete(
        DASHAVIDHA_QUESTION_MAP
    ) is True


def test_dashavidha_only_is_not_full_ayush_completion():
    answers = {
        question_id: "placeholder"
        for parameter, question_id in DASHAVIDHA_QUESTION_MAP.items()
        if parameter in DASHAVIDHA_ORDER
    }

    validator = AYUSHValidator(answers)

    assert validator.is_complete(
        DASHAVIDHA_QUESTION_MAP
    ) is False


# ============================================================
# 3. CONDITION MAP REGRESSION TESTS
# ============================================================

@pytest.fixture(scope="module")
def condition_map() -> ConditionMap:
    """The real map, not a stub.

    Ten tests below requested this fixture and nothing defined it, so all ten errored
    rather than ran - they were counted as neither passing nor failing, which is the
    quietest way for a test to stop testing.
    """
    from sanjeevani_ml.service.main import CONDITION_MAP_PATH

    return ConditionMap(str(CONDITION_MAP_PATH))


def test_condition_map_scores_are_clamped(condition_map):
    answers = {
        "question_a": "yes",
        "question_b": "yes",
    }

    scores = condition_map.calculate_plausibility(
        answers
    )

    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_unanswered_questions_do_not_add_evidence(
    condition_map,
):
    scores = condition_map.calculate_plausibility({})

    for score in scores.values():
        assert score == 0.0


def test_multiselect_matching(condition_map):
    assert condition_map._value_matches(
        ["a", "b"],
        "a",
    )

    assert condition_map._value_matches(
        ["a", "b"],
        "b",
    )

    assert not condition_map._value_matches(
        ["a", "b"],
        "c",
    )


def test_scalar_matching(condition_map):
    assert condition_map._value_matches(
        "yes",
        "yes",
    )

    assert not condition_map._value_matches(
        "yes",
        "no",
    )


def test_no_condition_evidence_preserves_original_order(
    condition_map,
):
    candidates = [
        "q1",
        "q2",
        "q3",
    ]

    ranked = condition_map.rank_questions(
        candidates=candidates,
        answers={},
    )

    assert ranked == candidates


# ============================================================
# 4. AYUSH MODE / ALLOPATHIC MODE ISOLATION
# ============================================================

def _engine():
    from sanjeevani_ml.dialogue.engine import DialogueEngine
    from sanjeevani_ml.service.main import CONDITION_MAP_PATH, ONTOLOGY_PATH

    return DialogueEngine(str(ONTOLOGY_PATH), condition_map_path=str(CONDITION_MAP_PATH))


#: A fresh engine per test: the engine holds per-session state, and a module-scoped one
#: would let one test's answers leak into the next.
@pytest.fixture()
def engine():
    return _engine()


@pytest.fixture()
def ayush_engine():
    return _engine()


def _walk(engine, session_id: str, mode: str) -> list[str]:
    """Every question id an interview asks, start to finish."""
    from sanjeevani_ml.schemas import Answer

    question = engine.start_interview(session_id=session_id, language="en", mode=mode)
    asked = []
    while question is not None and len(asked) < 80:
        asked.append(question.question_id)
        question = engine.submit_answer(
            session_id, Answer(question_id=question.question_id, value="no")
        )
    return asked


#: What an AYUSH question id actually looks like. The ontology uses `dp.` for the
#: ten-fold examination and `av.` for ahara-vihara — there is no `ayush.` prefix, so a
#: test asserting on one would pass against any interview at all.
AYUSH_PREFIXES = ("dp.", "av.")


def test_allopathic_mode_does_not_use_ayush_questions(engine):
    """An allopathic patient must never be asked about their prakriti.

    This previously stopped after one question and checked for an `ayush.` prefix that
    no question has, so it would have passed even if every Dashavidha parameter had
    been asked.
    """
    asked = _walk(engine, "test-allopathic-ayush-isolation", "allopathic")

    assert asked, "the allopathic interview asked nothing"
    leaked = [q for q in asked if q.startswith(AYUSH_PREFIXES)]
    assert not leaked, f"allopathic interview asked AYUSH questions: {leaked}"


def test_ayush_mode_can_enter_ayush_flow(ayush_engine):
    """An AYUSH interview must actually reach the AYUSH questions.

    The earlier version asserted the FIRST question started with `ayush.` - which no
    question does, and which the first question could not be anyway: the interview
    opens with the chief complaint regardless of mode.
    """
    asked = _walk(ayush_engine, "test-ayush-flow", "ayush")

    ayush_questions = [q for q in asked if q.startswith(AYUSH_PREFIXES)]
    assert ayush_questions, f"ayush mode asked no AYUSH questions; got {asked}"


# ============================================================
# 5. DASHAVIDHA ORDERING
# ============================================================

def test_dashavidha_order_is_preserved(ayush_engine):
    """The condition map must not reorder the ten-fold examination.

    Plausibility scoring exists to bring a relevant question forward. Dashavidha is not
    a pool to reorder - it is a sequence a practitioner performs in order, and the
    ontology declares that order. Reordering it would be the software quietly
    overruling the instrument.
    """
    asked = _walk(ayush_engine, "test-dashavidha-order", "ayush")

    observed = [q for q in asked if q.startswith("dp.")]
    expected = [
        DASHAVIDHA_QUESTION_MAP[parameter]
        for parameter in DASHAVIDHA_ORDER
        if DASHAVIDHA_QUESTION_MAP[parameter] in observed
    ]

    assert observed, "the AYUSH interview asked no Dashavidha questions"
    assert observed == expected, (
        f"Dashavidha was reordered. asked={observed} ontology={expected}"
    )


# ============================================================
# 6. AYUSH SHOULD NOT BE DIAGNOSIS
# ============================================================

def test_question_does_not_expose_diagnosis(
    ayush_engine,
):
    question = ayush_engine.start_interview(
        session_id="test-no-diagnosis",
        language="en",
        mode="ayush",
    )

    assert not hasattr(
        question,
        "diagnosis",
    )

    assert not hasattr(
        question,
        "plausibility",
    )


# ============================================================
# 7. RED FLAG DOES NOT TERMINATE INTERVIEW
# ============================================================

def test_red_flag_does_not_stop_interview(
    ayush_engine,
):
    """
    This test is structural.

    Your engine documentation states that a red flag
    should be recorded but should NOT terminate
    the interview.
    """

    session_id = "test-ayush-red-flag"

    question = ayush_engine.start_interview(
        session_id=session_id,
        language="en",
        mode="ayush",
    )

    assert question is not None

    # The exact red-flag injection depends on your
    # current redflag implementation.
    #
    # Once evaluate_answer() is wired into the test,
    # submit an answer that triggers a known test rule
    # and assert that another question is returned.


# ============================================================
# 8. AYUSH COMPLETENESS
# ============================================================

def test_full_ayush_interview_contains_all_required_fields():
    answers = make_answers()

    validator = AYUSHValidator(answers)

    missing_dashavidha = validator.missing_dashavidha(
        DASHAVIDHA_QUESTION_MAP
    )

    missing_additional = validator.missing_additional(
        DASHAVIDHA_QUESTION_MAP
    )

    assert missing_dashavidha == []
    assert missing_additional == []


# ============================================================
# 9. NO CLINICAL CONTENT IS GENERATED BY THE TEST
# ============================================================

def test_ayush_tests_use_no_unvalidated_clinical_scoring():
    """
    Safety boundary:

    This test suite validates the SOFTWARE STRUCTURE,
    not Ayurvedic clinical scoring.

    Actual Prakriti/Sara/Koshtha/etc. scoring must only
    be introduced after AIIA/practitioner validation.
    """

    answers = make_answers()

    assert all(
        value == "placeholder"
        for value in answers.values()
    )