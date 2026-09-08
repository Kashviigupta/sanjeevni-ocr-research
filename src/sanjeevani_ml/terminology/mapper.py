"""Map extracted entities to curated terminology codes — KASHVI owns this file.

Never invent a code. A curated table row is the only source of truth for a
code; no confident match means ``code=None`` plus the original text, which is
a correct answer, not a failure. The fuzzy-match threshold
(``settings.term_match_threshold``, currently 0.86) is deliberately strict:
"HDL" and "LDL" are one character apart and clinically opposite, and a wrong
LOINC/SNOMED/RxNorm code becomes a wrong code in a patient's permanent record.

This module is the seam between `extraction/` (shape only, no clinical
meaning) and `fhir/` (needs real codes to build valid Observation /
MedicationRequest / Condition resources) and `interactions/checker.py`
(needs `generic_name` and `normalize` to key its lookups on generics).
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from ..config import settings
from ..extraction.patterns import DRUG_LINE_RE
from ..schemas import ExtractedEntity, ExtractionResult, TerminologyCode
from .loader import load_conditions, load_drugs, load_lab_terms

__all__ = ["normalize", "generic_name", "code_result"]

log = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and collapse punctuation/whitespace, for comparing lookup keys.

    Used both here and in ``interactions/checker.py`` so that "Warfarin",
    "warfarin ", and "WARFARIN" all key to the same interaction-table row.
    """
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def generic_name(drug: str) -> str | None:
    """Brand or generic name in, generic name out.

    Indian prescriptions are written almost exclusively in brand names, but
    the interaction table and RxNorm coding are keyed on generics — skipping
    this step means the interaction checker silently finds nothing.

    Args:
        drug: A drug name as written on a prescription — brand or generic.

    Returns:
        The generic name exactly as it appears in ``drugs_seed.csv``, or
        ``None`` if nothing in the curated table matches. ``None`` is a
        correct, honest answer for a drug outside our ~300-drug table — the
        caller decides how to fall back (``checker.py`` falls back to the
        original spelling rather than dropping the drug from the check).
    """
    if not drug or not drug.strip():
        return None
    target = normalize(drug)
    for row in load_drugs():
        if normalize(row.get("generic", "")) == target:
            return row["generic"]
        brands = [b for b in row.get("brands", "").split("|") if b]
        if any(normalize(b) == target for b in brands):
            return row["generic"]
    return None


def _drug_terminology_code(drug_name: str) -> TerminologyCode | None:
    """RxNorm code for a resolved drug name, matched against generic or any brand."""
    target = normalize(drug_name)
    for row in load_drugs():
        brands = [b for b in row.get("brands", "").split("|") if b]
        if normalize(row.get("generic", "")) == target or any(normalize(b) == target for b in brands):
            return TerminologyCode(system="RxNorm", code=row["code"], display=row["display"])
    return None


def _best_match(
    term: str, rows: list[dict[str, str]], *, name_field: str, synonym_field: str = "synonyms"
) -> tuple[dict[str, str] | None, float]:
    """Find the best-scoring curated row for `term` among its name + synonyms.

    An exact match (after :func:`normalize`) short-circuits at score 1.0.
    Otherwise scores every candidate label with :class:`difflib.SequenceMatcher`
    and keeps the best. Deliberately simple and debuggable — no embeddings, no
    external service, so a judge can be shown exactly why a match did or did
    not happen.

    Returns:
        ``(best_row, best_score)``, or ``(None, 0.0)`` if `rows` is empty or
        `term` normalizes to nothing.
    """
    target = normalize(term)
    if not target:
        return None, 0.0

    best_row: dict[str, str] | None = None
    best_score = 0.0
    for row in rows:
        candidates = [row.get(name_field, "")]
        candidates += [s for s in row.get(synonym_field, "").split("|") if s]
        for candidate in candidates:
            candidate_norm = normalize(candidate)
            if not candidate_norm:
                continue
            if candidate_norm == target:
                return row, 1.0
            score = SequenceMatcher(None, target, candidate_norm).ratio()
            if score > best_score:
                best_score, best_row = score, row
    return best_row, best_score


def _code_lab_entity(entity: ExtractedEntity) -> ExtractedEntity:
    """Match a lab entity's analyte name (`entity.text`) against the LOINC table."""
    row, score = _best_match(entity.text, load_lab_terms(), name_field="term")
    if row is None or score < settings.term_match_threshold:
        return entity  # no confident match — code stays None, correctly
    code = TerminologyCode(system="LOINC", code=row["code"], display=row["display"])
    # Fill in a unit only if extraction didn't already capture one from the
    # document itself — the document's own unit always wins when present.
    unit = entity.unit or (row.get("unit") or None)
    return entity.model_copy(update={"code": code, "unit": unit})


def _code_diagnosis_entity(entity: ExtractedEntity) -> ExtractedEntity:
    """Match a diagnosis entity's condition text against the SNOMED table."""
    row, score = _best_match(entity.text, load_conditions(), name_field="term")
    if row is None or score < settings.term_match_threshold:
        return entity
    code = TerminologyCode(system="SNOMED", code=row["code"], display=row["display"])
    return entity.model_copy(update={"code": code})


def _code_medication_entity(entity: ExtractedEntity) -> ExtractedEntity:
    """Match a medication entity against the drugs table for an RxNorm code.

    `entity.text` for a medication is the raw prescription line plus a
    plain-language dose annotation (see ``extraction/structurer.py``), not a
    bare drug name — so the name is re-extracted with the same
    :data:`DRUG_LINE_RE` pattern the structurer used, rather than fuzzy-matching
    over the whole annotated string.
    """
    match = DRUG_LINE_RE.search(entity.text)
    name = match.group("name").strip() if match else entity.text
    code = _drug_terminology_code(name)
    if code is None:
        return entity
    return entity.model_copy(update={"code": code})


_CODER_BY_KIND = {
    "lab": _code_lab_entity,
    "diagnosis": _code_diagnosis_entity,
    "medication": _code_medication_entity,
}


def code_result(result: ExtractionResult) -> ExtractionResult:
    """Attach a terminology code to every entity with a confident curated-table match.

    Vitals and dates are never coded here — vitals get their LOINC codes at
    FHIR-build time (they map onto a small fixed set: BP, pulse, SpO2, etc.,
    not a lookup table), and a date has no terminology to resolve.

    Never invents a code: an entity below the fuzzy-match threshold, or with
    no candidate at all, keeps ``code=None`` and its original text untouched.
    That is a correct, honest outcome — a guessed code is a clinical safety
    bug, not a corner we're allowed to cut for the demo.

    Args:
        result: The output of ``extraction.structurer.structure()``.

    Returns:
        A new :class:`ExtractionResult` with the same entities, each newly
        carrying a code where one was confidently found.
    """
    coded_entities = [_CODER_BY_KIND.get(e.kind, lambda entity: entity)(e) for e in result.entities]
    matched = sum(1 for e in coded_entities if e.code is not None)
    # Counts only — never entity text, never a patient name.
    log.info("code_result(): %d/%d entities matched a terminology code", matched, len(coded_entities))
    return result.model_copy(update={"entities": coded_entities})