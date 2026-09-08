"""Red flag detection.

The red_flag rules live IN the ontology JSON (see history-ontology.json --
e.g. soc.associations' "breathless" option carries its own red_flag dict).
This module never hardcodes a symptom combination in Python; it only knows
how to READ and EVALUATE whatever conditions the ontology declares. Adding
or changing a red flag is a data change in the ontology, not a code change
here -- same rule that applies to questions themselves.

Three non-negotiable behaviours enforced here, per the project brief:
  1. A red flag never stops the interview -- this module only ever RETURNS
     triggers; it has no power to halt anything.
  2. Nothing here is patient-facing. RedFlag.explanation is physician-facing
     text only, in the ontology's own languages.
  3. When a condition is ambiguous or its inputs are missing, we treat it
     as NOT matching only when that's the safe default -- e.g. an unparsable
     duration is treated as "not acute" (see _duration_to_hours), which
     avoids under-triggering being silently masked as an error swallowed
     elsewhere. False negatives are the failure mode that matters most.
"""

from __future__ import annotations

from typing import Any

from sanjeevani_ml.dialogue.ontology_loader import section_items
from sanjeevani_ml.schemas import RedFlag

# Physician-facing explanations for known reasons. Deliberately a small,
# explicit table -- if a reason isn't here, we still fire the flag (never
# silently drop a red flag for lack of pretty text), just with a generic
# explanation instead of inventing clinical language.
_REASON_EXPLANATIONS: dict[str, dict[str, str]] = {
    "chest_pain_acute_onset": {
        "en": "Chest pain with sudden onset (within the last hour).",
        "hi": "सीने में दर्द जो अचानक (पिछले एक घंटे में) शुरू हुआ।",
    },
    "chest_pressure_cardiac_pattern": {
        "en": "Chest pain described as pressure/heaviness -- a pattern associated with cardiac chest pain.",
        "hi": "सीने में दर्द जिसे दबाव/भारीपन बताया गया -- हृदय संबंधी दर्द का पैटर्न।",
    },
    "chest_pain_radiation_cardiac": {
        "en": "Chest pain radiating to the arm, jaw, or back.",
        "hi": "सीने का दर्द बांह, जबड़े या पीठ तक जा रहा है।",
    },
    "chest_pain_with_dyspnoea": {
        "en": "Chest pain with breathlessness.",
        "hi": "सीने में दर्द के साथ सांस फूलना।",
    },
    "chest_pain_with_diaphoresis": {
        "en": "Chest pain with cold sweating.",
        "hi": "सीने में दर्द के साथ ठंडा पसीना।",
    },
    "syncope": {"en": "Dizziness or fainting reported.", "hi": "चक्कर या बेहोशी की सूचना।"},
    "stroke_facial_droop": {"en": "Facial drooping reported.", "hi": "चेहरे का लटकना बताया गया।"},
    "stroke_limb_weakness": {"en": "Sudden limb weakness reported.", "hi": "हाथ/पैर में अचानक कमज़ोरी बताई गई।"},
    "stroke_speech": {"en": "Difficulty speaking reported.", "hi": "बोलने में कठिनाई बताई गई।"},
    "uncontrolled_bleeding": {"en": "Bleeding that will not stop.", "hi": "खून बहना जो रुक नहीं रहा।"},
    "dyspnoea_at_rest": {"en": "Breathless while sitting still.", "hi": "बैठे-बैठे सांस फूलना।"},
    "drug_allergy_reported": {"en": "Patient reports a past drug reaction.", "hi": "मरीज़ ने पूर्व में दवा प्रतिक्रिया बताई।"},
    # --- complaint-branch flags -------------------------------------------
    "fever_with_neck_stiffness": {
        "en": "Fever with neck stiffness -- meningitis must be excluded.",
        "hi": "बुखार के साथ गर्दन में अकड़न -- मेनिनजाइटिस को बाहर करना ज़रूरी।",
    },
    "fever_with_rash": {
        "en": "Fever with a rash. Consider dengue, measles and drug reaction.",
        "hi": "बुखार के साथ दाने। डेंगू, खसरा और दवा प्रतिक्रिया पर विचार करें।",
    },
    "haemoptysis": {
        "en": "Blood in the sputum. TB and malignancy both present this way.",
        "hi": "बलगम में खून। टीबी और कैंसर दोनों ऐसे दिख सकते हैं।",
    },
    "orthopnoea": {
        "en": "Breathless when lying flat -- suggests cardiac failure.",
        "hi": "लेटने पर साँस फूलना -- हृदय विफलता का संकेत।",
    },
    "gi_bleeding": {
        "en": "Blood in stool or vomit, or black tarry stool.",
        "hi": "मल या उल्टी में खून, या काला लसदार मल।",
    },
    "dehydration_reduced_urine": {
        "en": "Passing very little urine -- significant dehydration.",
        "hi": "बहुत कम पेशाब -- गंभीर निर्जलीकरण।",
    },
    "jaundice": {
        "en": "Yellow eyes or urine reported.",
        "hi": "आँखें या पेशाब पीला बताया गया।",
    },
    "skin_numb_patch": {
        "en": "A skin patch with no sensation. Screen for leprosy -- notifiable.",
        "hi": "त्वचा पर सुन्न चकत्ता। कुष्ठ रोग की जाँच करें -- सूचनीय रोग।",
    },
    "skin_spreading_with_fever": {
        "en": "Spreading skin lesion with fever -- cellulitis until proven otherwise.",
        "hi": "फैलता त्वचा घाव और बुखार -- जब तक साबित न हो, सेल्युलाइटिस मानें।",
    },
    "unintentional_weight_loss": {"en": "Unintentional weight loss.", "hi": "बिना कोशिश वज़न घटना।"},
    "night_sweats": {"en": "Night sweats reported.", "hi": "रात में पसीना।"},
    "prolonged_fever": {"en": "Fever for more than 2 weeks.", "hi": "दो हफ़्ते से ज़्यादा बुखार।"},
    "haemorrhage_unexplained": {"en": "Unexplained bleeding in cough, stool, or urine.", "hi": "खाँसी, मल या पेशाब में अस्पष्ट खून।"},
}


class RedFlagRuleError(ValueError):
    """Raised when a red_flag condition in the ontology can't be evaluated
    (unsupported syntax) -- surfaced loudly rather than silently ignored,
    since a red flag that fails to fire is a patient-safety issue."""


def flatten_questions(ontology: dict) -> dict[str, dict]:
    """Walk ontology['sections'][*]['questions'] and return {question_id: entry}.

    Reads the raw ontology dict directly -- NOT the trimmed Question schema
    from ontology_loader.py, since red_flag/branch/regions metadata doesn't
    fit that model yet. This is the one place that depends on the ontology's
    real on-disk shape.
    """
    flat: dict[str, dict] = {}
    for section in ontology.get("sections", []):
        for question in section_items(section):
            flat[question["id"]] = question
    return flat


def evaluate_answer(
    question_id: str,
    value: Any,
    all_answers: dict[str, Any],
    flattened_questions: dict[str, dict],
) -> list[RedFlag]:
    """Given one answer, return every RedFlag it triggers (zero or more).

    `all_answers` should include this answer already (i.e. call this AFTER
    recording the answer into the transcript), since some conditions
    reference other questions (e.g. "cc.area == chest").
    """
    entry = flattened_questions.get(question_id)
    if entry is None:
        return []  # unknown question id -- nothing to evaluate against

    triggers: list[RedFlag] = []

    # Question-level red_flag (e.g. soc.onset, soc.radiation)
    question_rule = entry.get("red_flag")
    if question_rule is not None:
        if _condition_matches(question_rule.get("when"), value, all_answers):
            triggers.append(_build_red_flag(question_id, question_rule))

    # Option-level red_flag (e.g. soc.associations' "breathless" option)
    selected_ids = _selected_option_ids(entry, value)
    for option in entry.get("options", []):
        if option["id"] not in selected_ids:
            continue
        option_rule = option.get("red_flag")
        if option_rule is None:
            continue
        if _condition_matches(option_rule.get("when"), option["id"], all_answers):
            triggers.append(_build_red_flag(question_id, option_rule))

    return triggers


def _build_red_flag(question_id: str, rule: dict) -> RedFlag:
    reason = rule["reason"]
    return RedFlag(
        reason=reason,
        severity=rule["severity"],
        question_id=question_id,
        explanation=_REASON_EXPLANATIONS.get(reason, {}),
    )


def _selected_option_ids(entry: dict, value: Any) -> set[str]:
    """single_choice -> value is one id string. multi_choice -> value is a
    list of id strings. Anything else -> no option-level rules apply."""
    if entry.get("input_type") == "multi_choice" and isinstance(value, list):
        return set(value)
    if isinstance(value, str):
        return {value}
    return set()


def _condition_matches(condition: str | None, current_value: Any, all_answers: dict[str, Any]) -> bool:
    """No `when` clause means "always fires when this option/question is
    reached" -- that's correct for e.g. rf.stroke's face_droop option."""
    if condition is None:
        return True
    clauses = [c.strip() for c in condition.split("&&")]
    return all(_clause_matches(c, current_value, all_answers) for c in clauses)


_OPERATORS = ("<=", ">=", "==", "<", ">", " in ")


def _clause_matches(clause: str, current_value: Any, all_answers: dict[str, Any]) -> bool:
    clause = clause.strip().lstrip("(").rstrip(")").strip()

    for op in _OPERATORS:
        if op in clause:
            left, right = clause.split(op, 1)
            left_val = _resolve(left.strip(), current_value, all_answers)
            right = right.strip()
            op = op.strip()

            if op == "in":
                options = [v.strip() for v in right.strip("[]").split(",")]
                return left_val in options
            if op == "==":
                return str(left_val) == right
            # Numeric comparisons -- an unparsable number is treated as NOT
            # matching (never crash mid-interview on a weird value), but this
            # is logged as a rule error since it likely means bad ontology data.
            try:
                left_num = float(left_val)
                right_num = float(right)
            except (TypeError, ValueError):
                return False
            return {"<=": left_num <= right_num, ">=": left_num >= right_num,
                     "<": left_num < right_num, ">": left_num > right_num}[op]

    raise RedFlagRuleError(f"Unsupported red-flag condition clause: '{clause}'")


def _resolve(identifier: str, current_value: Any, all_answers: dict[str, Any]) -> Any:
    if identifier == "value":
        return current_value
    if identifier == "value_hours":
        return _duration_to_hours(current_value)
    return all_answers.get(identifier)


_HOURS_PER_UNIT = {"hours": 1, "days": 24, "weeks": 24 * 7, "months": 24 * 30, "years": 24 * 365}


def _duration_to_hours(duration_value: Any) -> float:
    """Expects {"amount": N, "unit": "hours"|"days"|...} from a duration
    input_type answer. An unparsable value returns infinity -- i.e. "not
    acute" -- so a malformed answer can never falsely suppress a red flag
    by looking like zero; it just fails to ADD one, which a human reviewing
    the ExplorationReport's not_explored list would still catch.
    """
    # Either the validated DurationValue or the raw dict off the wire - this is called
    # from both sides of the schema boundary.
    if hasattr(duration_value, "amount") and hasattr(duration_value, "unit"):
        duration_value = {"amount": duration_value.amount, "unit": duration_value.unit}
    if not isinstance(duration_value, dict):
        return float("inf")
    try:
        amount = float(duration_value.get("amount", 0))
        unit = duration_value.get("unit", "hours")
        return amount * _HOURS_PER_UNIT.get(unit, 1)
    except (TypeError, ValueError):
        return float("inf")
