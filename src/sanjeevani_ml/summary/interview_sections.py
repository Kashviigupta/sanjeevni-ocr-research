"""Interview-derived summary sections, built against the REAL question ids from
shared/clinical/history-ontology.json (the version reviewed on 2026-08-22) —
not guessed keys.

Every function reads `transcript.answers[question_id]` defensively: a missing
key means that question was skipped or not reached (branching interview — not
every patient sees every question), and produces an honest gap note rather than
a crash or a fabricated default.

Known limitation, stated rather than hidden: the ontology is an explicit SEED.
Sections 2 ("socrates") only fully covers the "pain" branch of chief complaint;
the fever/respiratory/digestive/constitutional/dermatology branches are named as
`branch` targets in `cc.primary` but their question sets are not in the ontology
snapshot this module was built against. HPI for a non-pain complaint therefore
falls back to `cc.describe` (the patient's own free narration) plus a note that
structured branch questions are pending. This is a real, temporary content gap,
not a bug — the fix is adding those branches to the ontology, not to this module.

`ayurvedic_assessment` (Dashavidha Pariksha) is deliberately NOT built here even
though its question ids are known: `dp.prakriti` and `dp.vikriti` are marked
`"input_type": "derived"` in the ontology (computed by Mahek's dialogue engine
from a sub-questionnaire this module has no access to), and the ontology's own
`$reference` note requires every parameter to be validated with an AIIA
practitioner before use. Rendering it from raw answer ids without that
validation is exactly the shortcut the platform's AYUSH rule forbids.
"""
from __future__ import annotations

from typing import Any

from sanjeevani_ml.schemas import InterviewTranscript, SummarySection
from sanjeevani_ml.summary.ontology_labels import (
    BODY_REGION_LABELS,
    CC_PRIMARY_LABELS,
    PF_FAMILY_LABELS,
    PF_HABITS_LABELS,
    PX_CONDITION_LABELS,
    ROS_SCREEN_LABELS,
    SOC_ASSOCIATION_LABELS,
    SOC_CHARACTER_LABELS,
    SOC_MODIFIER_LABELS,
)

_NOT_ANSWERED_EN = "Not answered or not reached in this interview."
_NOT_ANSWERED_HI = "इस साक्षात्कार में उत्तर नहीं दिया गया या यह प्रश्न नहीं आया।"


def _get(answers: dict[str, Any], question_id: str) -> Any:
    return answers.get(question_id)


def _labelled(value: Any, labels: dict[str, str]) -> str:
    """One selected option id -> its ontology label, or the raw id if unknown."""
    return labels.get(value, str(value)) if value is not None else ""


def _labelled_list(values: Any, labels: dict[str, str]) -> list[str]:
    if not isinstance(values, list):
        return []
    return [labels.get(v, str(v)) for v in values if v != "none"]


def build_chief_complaint_section(transcript: InterviewTranscript) -> SummarySection:
    """cc.area + cc.primary + cc.describe -> the "what brings you here" section.

    Leads with the patient's own words (`cc.describe`) when present, per the
    ontology's own note that it is "the single most valuable question in the
    interview" — the structured answers frame it, they do not replace it.
    """
    answers = transcript.answers
    area = _labelled(_get(answers, "cc.area"), BODY_REGION_LABELS)
    primary = _labelled(_get(answers, "cc.primary"), CC_PRIMARY_LABELS)
    narration = _get(answers, "cc.describe")

    if not any([area, primary, narration]):
        return SummarySection(
            heading="chief_complaint",
            content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )

    lines = []
    if primary:
        location = f" in the {area}" if area else ""
        lines.append(f"Presenting complaint: {primary}{location}.")
    if narration:
        lines.append(f'Patient\'s own words: "{narration}"')

    return SummarySection(
        heading="chief_complaint",
        content={"en": "\n".join(lines), "hi": "\n".join(lines)},  # verbatim narration untranslated
        provenance=["interview:cc.area", "interview:cc.primary", "interview:cc.describe"],
        confidence=1.0 if narration else 0.7,  # verbatim patient speech is the strongest signal we have
    )


def build_hpi_section(transcript: InterviewTranscript) -> SummarySection:
    """SOCRATES (pain branch) -> HPI. Falls back to free narration for any other
    chief complaint, since only the pain branch's question set exists yet.
    """
    answers = transcript.answers
    primary = _get(answers, "cc.primary")

    if primary != "pain":
        narration = _get(answers, "cc.describe")
        if narration:
            return SummarySection(
                heading="hpi",
                content={
                    "en": f'History of presenting illness, in the patient\'s own words: "{narration}". '
                          f'Structured follow-up questions for this type of complaint are not yet in '
                          f'the interview ontology.',
                    "hi": _NOT_ANSWERED_HI,
                },
                provenance=["interview:cc.describe"],
                confidence=0.5,
            )
        return SummarySection(
            heading="hpi", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )

    onset = _get(answers, "soc.onset")
    character = _labelled(_get(answers, "soc.character"), SOC_CHARACTER_LABELS)
    severity = _get(answers, "soc.severity")
    associations = _labelled_list(_get(answers, "soc.associations"), SOC_ASSOCIATION_LABELS)
    modifiers = _labelled_list(_get(answers, "soc.modifiers"), SOC_MODIFIER_LABELS)

    parts = []
    if onset:
        parts.append(f"Onset: {onset}.")
    if character:
        parts.append(f"Character: {character}.")
    if severity is not None:
        parts.append(f"Severity: {severity}/10.")
    if associations:
        parts.append(f"Associated with: {', '.join(associations)}.")
    if modifiers:
        parts.append(f"Modifiers: {', '.join(modifiers)}.")

    if not parts:
        return SummarySection(
            heading="hpi", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )

    text = " ".join(parts)
    return SummarySection(
        heading="hpi", content={"en": text, "hi": text},
        provenance=[f"interview:{k}" for k in ("soc.onset", "soc.character", "soc.severity",
                                                 "soc.associations", "soc.modifiers")],
        confidence=0.9,
    )


def build_past_medical_section(transcript: InterviewTranscript) -> SummarySection:
    """px.conditions -> known prior diagnoses, patient-reported."""
    conditions = _labelled_list(_get(transcript.answers, "px.conditions"), PX_CONDITION_LABELS)
    if not conditions:
        return SummarySection(
            heading="past_medical", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )
    text = "Patient-reported known conditions: " + ", ".join(conditions) + "."
    return SummarySection(
        heading="past_medical", content={"en": text, "hi": text},
        provenance=["interview:px.conditions"], confidence=0.85,
    )


def build_past_surgical_section(transcript: InterviewTranscript) -> SummarySection:
    """px.surgery (yes/no) + px.surgery_detail (free text) -> surgical history."""
    had_surgery = _get(transcript.answers, "px.surgery")
    detail = _get(transcript.answers, "px.surgery_detail")

    if had_surgery is None:
        return SummarySection(
            heading="past_surgical", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )
    if had_surgery in (False, "no"):
        text = "Patient reports no prior operations."
        return SummarySection(
            heading="past_surgical", content={"en": text, "hi": text},
            provenance=["interview:px.surgery"], confidence=0.9,
        )
    text = f"Patient reports a prior operation: \"{detail}\"." if detail else \
        "Patient reports a prior operation; details not captured."
    return SummarySection(
        heading="past_surgical", content={"en": text, "hi": text},
        provenance=["interview:px.surgery", "interview:px.surgery_detail"], confidence=0.8,
    )


def build_family_section(transcript: InterviewTranscript) -> SummarySection:
    """pf.family -> family history of the four screened conditions."""
    conditions = _labelled_list(_get(transcript.answers, "pf.family"), PF_FAMILY_LABELS)
    if not conditions:
        return SummarySection(
            heading="family", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )
    text = "Family history: " + ", ".join(conditions) + "."
    return SummarySection(
        heading="family", content={"en": text, "hi": text},
        provenance=["interview:pf.family"], confidence=0.85,
    )


def build_personal_section(transcript: InterviewTranscript) -> SummarySection:
    """pf.habits -> social/personal history (smoking, tobacco, alcohol)."""
    habits = _labelled_list(_get(transcript.answers, "pf.habits"), PF_HABITS_LABELS)
    if not habits:
        return SummarySection(
            heading="personal", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )
    text = "Reported habits: " + ", ".join(habits) + "."
    return SummarySection(
        heading="personal", content={"en": text, "hi": text},
        provenance=["interview:pf.habits"], confidence=0.85,
    )


def build_ros_section(transcript: InterviewTranscript) -> SummarySection:
    """ros.screen + ros.anything -> the short review-of-systems screen."""
    screen = _labelled_list(_get(transcript.answers, "ros.screen"), ROS_SCREEN_LABELS)
    anything_else = _get(transcript.answers, "ros.anything")

    lines = []
    if screen:
        lines.append("Reported in the last few weeks: " + ", ".join(screen) + ".")
    if anything_else:
        lines.append(f'Patient added: "{anything_else}"')

    if not lines:
        return SummarySection(
            heading="ros", content={"en": _NOT_ANSWERED_EN, "hi": _NOT_ANSWERED_HI},
            provenance=[], confidence=0.0,
        )
    text = " ".join(lines)
    return SummarySection(
        heading="ros", content={"en": text, "hi": text},
        provenance=["interview:ros.screen", "interview:ros.anything"], confidence=0.8,
    )


def build_ayurvedic_assessment_placeholder() -> SummarySection:
    """Deliberately a gap section. See module docstring for why."""
    return SummarySection(
        heading="ayurvedic_assessment",
        content={
            "en": "Dashavidha Pariksha results require AIIA practitioner-reviewed "
                  "content mapping and are not rendered by this module. See the raw "
                  "interview answers for dp.* fields.",
            "hi": "दशविध परीक्षा के परिणामों के लिए AIIA चिकित्सक द्वारा समीक्षित सामग्री "
                  "मानचित्रण आवश्यक है और यह मॉड्यूल इसे प्रस्तुत नहीं करता।",
        },
        provenance=[], confidence=0.0,
    )


def build_interview_sections(transcript: InterviewTranscript) -> list[SummarySection]:
    """Every interview-derived section this module can build, in ontology order."""
    return [
        build_chief_complaint_section(transcript),
        build_hpi_section(transcript),
        build_past_medical_section(transcript),
        build_past_surgical_section(transcript),
        build_family_section(transcript),
        build_personal_section(transcript),
        build_ros_section(transcript),
        build_ayurvedic_assessment_placeholder(),
    ]
