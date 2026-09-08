"""Detect a disagreement between what the patient SAID and what a document SHOWS,
scoped to medications only for now — the exact example the problem statement gives
("patient reports no medicines" against a scanned prescription).

Primary signal is the REAL ontology question `rx.current` ("Are you taking any
medicines right now?", yes_no) — a structured answer is more reliable than
scanning free text. Falls back to keyword-matching `narration` only when
`rx.current` was skipped or not reached — a branching interview does not
guarantee every patient sees every question, and a silent gap here would mean
this module simply stops checking without saying so.
"""
from __future__ import annotations

import re

from sanjeevani_ml.schemas import Contradiction, DocumentBundle, InterviewTranscript
from sanjeevani_ml.timeline.dedup import primary_drug_name

#: Fallback only — used when the structured `rx.current` answer is missing.
_NO_MEDICATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bno medicin",
        r"\bnot (on|taking) any medic",
        r"\bdon'?t take any medic",
        r"\bno tablets?\b",
        r"\bkoi dawa nahi\b",       # "no medicine" — common transliterated Hindi
    ]
]


def _patient_denies_medications_structured(transcript: InterviewTranscript) -> str | None:
    """The real ontology signal: `rx.current` == a falsy/"no" answer.

    Returns a human-readable description of the denial (for quoting in the
    Contradiction), or None if the patient said yes. Distinct from "the
    question was never reached" — see `detect_medication_contradictions`,
    which checks for the key's presence separately so an explicit "yes" is
    never treated the same as "no data".
    """
    answer = transcript.answers.get("rx.current")
    if answer in (False, "no", "no_tell"):
        return "Patient answered 'No' to \"Are you taking any medicines right now?\""
    return None


def _patient_denies_medications_narration(narration: list[str]) -> str | None:
    """Fallback for when `rx.current` was skipped: scan free narration instead."""
    for line in narration:
        if any(pattern.search(line) for pattern in _NO_MEDICATION_PATTERNS):
            return line
    return None


def _documented_medications(bundle: DocumentBundle) -> list[tuple[str, str | None]]:
    """(drug name, source document id) for every medication found in the documents."""
    found: list[tuple[str, str | None]] = []
    for entry in bundle.timeline:
        for entity in entry.entities:
            if entity.kind != "medication":
                continue
            name = primary_drug_name(entity.text) or entity.text
            found.append((name, entry.document_id))
    return found


def detect_medication_contradictions(
    transcript: InterviewTranscript, bundle: DocumentBundle
) -> list[Contradiction]:
    """A patient's denial of medication use, set against any medication a document shows.

    Checks the structured `rx.current` answer first (the real ontology signal);
    falls back to narration keyword-matching ONLY if that question's key is
    entirely absent from `answers` (never reached / skipped) — an explicit
    "yes" is never overridden by a stray narration phrase, which is exactly
    the bug a naive "check structured, then check narration regardless"
    ordering would produce.

    Returns one Contradiction per documented drug found alongside a denial —
    not one for the whole bundle — so the physician can open the exact scan
    that conflicts with what the patient said.

    Never invents a contradiction from a documented medication alone: silence
    (the question wasn't asked, or was answered "yes") is not evidence of
    anything. Only an explicit denial triggers a check.
    """
    if "rx.current" in transcript.answers:
        denial = _patient_denies_medications_structured(transcript)
    else:
        denial = _patient_denies_medications_narration(transcript.narration)
    if denial is None:
        return []

    documented = _documented_medications(bundle)
    return [
        Contradiction(
            topic="current medications",
            from_interview=denial,
            from_documents=f"Prescription found for {drug_name}",
            document_id=document_id,
        )
        for drug_name, document_id in documented
    ]
