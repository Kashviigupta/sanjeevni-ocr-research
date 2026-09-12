"""Turn an :class:`OcrResult` into an :class:`ExtractionResult`.

This is the ``structure()`` step of the pipeline described in the project brief:

    OcrResult  →  structure()  →  ExtractedEntity[]  →  code_result()  →  build()

``structure()`` returns a full :class:`ExtractionResult` (not a bare entity list):
that is the shape ``ml/tests/test_pipeline.py`` — the regression gate — expects,
and it is also what lets a caller ask "what kind of document is this, what date
was it, which facility issued it" without re-deriving that from the entity list
every time.

No entity produced here carries a terminology code (``code`` is always
``None``). Coding is ``terminology/mapper.py``'s job, and the never-invent-a-code
rule lives there, not here. This module's job is: find the clinical facts a
document contains, describe them in the way the patient will actually read them,
and say honestly how sure the *shape match* was.

Confidence discipline: every entity's confidence is
``ocr.mean_confidence * RULE_CONFIDENCE[rule]`` — OCR's own uncertainty
multiplied by how certain the matching rule is, assuming the characters were
read correctly. We never invent a number and never round it up.
"""

from __future__ import annotations

import logging
import re

from sanjeevani_ml.extraction.patterns import (
    BP_RE,
    DATE_DMY_RE,
    DIAGNOSIS_LINE_RE,
    DOSE_SLOT_RE,
    DRUG_LINE_RE,
    DRUG_LINE_REVERSED_RE,
    DURATION_RE,
    FACILITY_RE,
    FREQUENCY_RE,
    HEIGHT_RE,
    LAB_ROW_RE,
    PULSE_RE,
    RESP_RATE_RE,
    RULE_CONFIDENCE,
    SPO2_RE,
    STRENGTH_RE,
    TEMPERATURE_RE,
    WEIGHT_RE,
    normalize_text,
    parse_date,
    parse_dose_notation,
    split_conditions,
)
from sanjeevani_ml.schemas import ExtractedEntity, ExtractionResult, OcrResult, RecordType

__all__ = ["structure", "extract_dates"]

logger = logging.getLogger(__name__)


def _confidence(ocr_mean: float, rule_key: str) -> float:
    """OCR confidence x rule certainty, clipped to the [0, 1] the schema requires.

    Never returns a value higher than either input — that is the whole point of
    multiplying rather than picking the max.
    """
    raw = ocr_mean * RULE_CONFIDENCE[rule_key]
    return min(1.0, max(0.0, raw))


# ---------------------------------------------------------------------------
# Dates — public, standalone (test_pipeline.py imports this directly)
# ---------------------------------------------------------------------------

# Words that suggest a nearby date is *the* date of the visit/report, as opposed
# to some other number that merely looks like a date. Checked in a small window
# before the match. Deliberately excludes "DOB"/"birth" — a date of birth is not
# a clinical/visit date and must never be picked as one.
_DATE_LABEL_RE = re.compile(
    r"(date|collected|reported|issued|dated|दिनांक|तारीख)\s*[:\-]?\s*$", re.I
)
_DATE_LABEL_WINDOW = 20  # chars of context checked before a date match


def extract_dates(text: str) -> tuple[str | None, float]:
    """Find the single best clinical date in a document.

    Prefers a date immediately preceded by a label such as "Date:",
    "Collected on:", "Reported:", or "दिनांक:" over an unlabelled one, and an
    unambiguous DD/MM reading (day > 12) over one where the order had to be
    assumed. Among equally-preferred candidates, the first in reading order
    wins.

    Args:
        text: Document text (raw or already normalized; this function
            normalizes internally so either works).

    Returns:
        ``(iso_date, confidence)`` — ``iso_date`` is ``None`` when nothing in
        the text parses as a real calendar date, in which case confidence is
        ``0.0``. Confidence here is the *rule's* certainty only (not
        multiplied by OCR confidence) — callers combine it with
        ``ocr.mean_confidence`` themselves.
    """
    normalized = normalize_text(text)
    best: tuple[bool, bool, "object"] | None = None  # (is_labelled, is_unambiguous, parsed)
    best_parsed = None

    for match in DATE_DMY_RE.finditer(normalized):
        parsed = parse_date(match)
        if parsed is None:
            continue  # not a real calendar date — never guessed at, just skipped
        context = normalized[max(0, match.start() - _DATE_LABEL_WINDOW) : match.start()]
        is_labelled = _DATE_LABEL_RE.search(context) is not None
        candidate_rank = (is_labelled, not parsed.ambiguous)
        if best is None or candidate_rank > best[:2]:
            best = (*candidate_rank, None)
            best_parsed = parsed

    if best_parsed is None:
        return None, 0.0
    return best_parsed.value.isoformat(), best_parsed.rule_confidence


# ---------------------------------------------------------------------------
# Dosing, expanded to plain language
# ---------------------------------------------------------------------------

# 1-0-1 conventionally means morning / afternoon / night in Indian prescribing.
# A 4th slot, when present, is a bedtime dose added on top of that.
_SLOT_LABELS: tuple[str, ...] = ("morning", "afternoon", "night", "bedtime")
_AT_LABELS = {"night", "bedtime"}  # "at night" / "at bedtime" vs "in the morning"

_FREQUENCY_TEXT: dict[str, str] = {
    "OD": "once daily",
    "BD": "twice daily",
    "BID": "twice daily",
    "TDS": "three times daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "QDS": "four times daily",
    "HS": "at bedtime",
    "SOS": "as needed",
    "PRN": "as needed",
    "STAT": "immediately",
}


def _format_slot(value: float) -> str:
    """Render a dose-schedule slot the way it was written: "1" not "1.0"."""
    return str(int(value)) if value == int(value) else str(value)


def _dose_schedule_text(schedule) -> str:
    """Turn a parsed dose schedule into a sentence a patient can read.

    "1-0-1" -> "1 tablet in the morning and 1 tablet at night". This is the
    whole point of treating Indian dosing notation as first-class: the raw
    notation means nothing to most patients, the plain-language version does.
    """
    segments: list[str] = []
    for label, amount in zip(_SLOT_LABELS, schedule.slots):
        if not amount:
            continue
        if amount == 0.5:
            quantity = "half a tablet"
        else:
            formatted = _format_slot(amount)
            quantity = f"{formatted} tablet" if formatted == "1" else f"{formatted} tablets"
        connector = "at" if label in _AT_LABELS else "in the"
        segments.append(f"{quantity} {connector} {label}")

    if not segments:
        return ""
    if len(segments) == 1:
        return segments[0]
    return ", ".join(segments[:-1]) + " and " + segments[-1]


def _frequency_text(freq_token: str) -> str:
    return _FREQUENCY_TEXT.get(freq_token.upper(), freq_token.upper())


# ---------------------------------------------------------------------------
# Structuring
# ---------------------------------------------------------------------------


def structure(ocr: OcrResult, hint: RecordType | None = None) -> ExtractionResult:
    """Extract every clinical fact this module recognises out of OCR text.

    Every pattern is anchored specifically enough (a dosage-form prefix, a
    vitals label, a full-line lab-row shape) that they do not compete for the
    same text, so extraction order does not encode precedence between kinds.
    Where a document is ambiguous in a way we cannot resolve safely (an
    unclear date order, a facility line we're not sure about), the resulting
    entity is emitted anyway with a correspondingly lower confidence rather
    than being silently dropped or guessed.

    Never raises on unparseable or empty input — degrading to an empty,
    low-confidence result (with a warning) is the correct behaviour so the app
    can fall back to manual entry, per the platform's degrade-never-fail
    principle.

    Args:
        ocr: Mahek's OCR output — raw text plus its own mean confidence.
        hint: What the patient said this document is (they tapped "this is a
            lab report" before scanning). Overrides the entity-shape heuristic
            in `_classify` when given — a patient's direct statement about
            their own document is a stronger, cheaper signal than anything we
            can infer from what patterns happened to match, and discarding it
            silently is worse than not asking for it at all.

    Returns:
        An :class:`ExtractionResult` with ``code=None`` on every entity and an
        empty ``fhir`` — coding and FHIR building are separate steps.
    """
    text = normalize_text(ocr.text)
    ocr_mean = ocr.mean_confidence
    warnings: list[str] = list(ocr.warnings)

    entities: list[ExtractedEntity] = []
    entities.extend(_extract_date_entities(text, ocr_mean))
    entities.extend(_extract_diagnoses(text, ocr_mean))
    entities.extend(_extract_drug_lines(text, ocr_mean))
    entities.extend(_extract_lab_rows(text, ocr_mean))
    entities.extend(_extract_vitals(text, ocr_mean))

    facility_entity = _facility_entity(text, ocr_mean)
    if facility_entity is not None:
        entities.append(facility_entity)

    suggested_type = hint or _classify(entities)
    clinical_date, _date_rule_confidence = extract_dates(text)

    if not text.strip():
        warnings.append("No text was extracted from the document.")
    elif not entities:
        warnings.append(
            "No recognizable clinical entities were found; this document may need manual entry."
        )

    mean_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.0

    result = ExtractionResult(
        entities=entities,
        suggested_type=suggested_type,
        suggested_title=_suggest_title(suggested_type, facility_entity),
        clinical_date=clinical_date,
        facility=facility_entity.text if facility_entity is not None else None,
        fhir={},
        mean_confidence=mean_confidence,
        engine=ocr.engine,
        warnings=warnings,
    )

    # Counts and confidence only — never the extracted text, never a patient name.
    logger.info(
        "structure(): %d entities, type=%s, engine=%s, mean_confidence=%.2f, hint=%s",
        len(entities),
        suggested_type,
        ocr.engine,
        mean_confidence,
        hint,
    )
    return result


def _classify(entities: list[ExtractedEntity]) -> str:
    """Pick the document type the entity mix most looks like.

    Only consulted when the caller gave no `hint` — see `structure()`. A
    patient's own statement about their document always wins over this
    heuristic; this function exists for the (common) case where none was
    given.

    Medications beat labs beat a bare diagnosis, because a document with any
    prescribed drugs on it is functionally a prescription even if it also
    mentions a diagnosis; a document with lab rows and no drugs is a lab
    report. Falls back to "note" rather than guessing at "condition" for a
    lone diagnosis line, since we'd rather under-classify than mis-classify.
    """
    kinds = {e.kind for e in entities}
    if "medication" in kinds:
        return "prescription"
    if "lab" in kinds:
        return "lab"
    if "diagnosis" in kinds:
        return "condition"
    return "note"


def _suggest_title(suggested_type: str, facility_entity: ExtractedEntity | None) -> str:
    if facility_entity is None:
        return "Untitled document"
    label = {"prescription": "Prescription", "lab": "Lab Report", "condition": "Diagnosis"}.get(
        suggested_type, "Document"
    )
    return f"{label} — {facility_entity.text}"


# ---------------------------------------------------------------------------
# Per-occurrence date entities
# ---------------------------------------------------------------------------


def _extract_date_entities(text: str, ocr_mean: float) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for match in DATE_DMY_RE.finditer(text):
        parsed = parse_date(match)
        if parsed is None:
            continue  # not a real calendar date — dropped, never repaired
        note = " (day/month order assumed, please confirm)" if parsed.ambiguous else ""
        entities.append(
            ExtractedEntity(
                kind="date",
                text=f"{parsed.raw}{note}",
                value=parsed.value.isoformat(),
                confidence=_confidence(
                    ocr_mean, "date_assumed_dmy" if parsed.ambiguous else "date_unambiguous"
                ),
            )
        )
    return entities


# ---------------------------------------------------------------------------
# Facility
# ---------------------------------------------------------------------------


def _facility_entity(text: str, ocr_mean: float) -> ExtractedEntity | None:
    # Only the first match: a document has one issuing facility, usually in the
    # header, and later matches are more likely to be noise (e.g. a referral
    # mention) than the actual performer/organisation.
    match = FACILITY_RE.search(text)
    if match is None:
        return None
    return ExtractedEntity(
        kind="facility",
        text=match.group("facility").strip(),
        confidence=_confidence(ocr_mean, "facility"),
    )


# ---------------------------------------------------------------------------
# Diagnoses
# ---------------------------------------------------------------------------


def _extract_diagnoses(text: str, ocr_mean: float) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for match in DIAGNOSIS_LINE_RE.finditer(text):
        for condition in split_conditions(match.group("conditions")):
            entities.append(
                ExtractedEntity(
                    kind="diagnosis",
                    text=condition,
                    confidence=_confidence(ocr_mean, "diagnosis_line"),
                )
            )
    return entities


# ---------------------------------------------------------------------------
# Prescription / drug lines
# ---------------------------------------------------------------------------


def _extract_drug_lines(text: str, ocr_mean: float) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    # Two passes: the usual "Tab Glycomet 500mg" order, then the reversed
    # "Electral powder" order some common OTC items use. Two independent
    # loops rather than one merged pattern, per the reasoning in patterns.py.
    for match in DRUG_LINE_REVERSED_RE.finditer(text):
        name = match.group("name").strip()
        raw_line = match.group(0).strip()
        entities.append(
            ExtractedEntity(
                kind="medication",
                text=raw_line,
                value=name,
                unit=None,
                confidence=_confidence(ocr_mean, "drug_line"),
            )
        )
    for match in DRUG_LINE_RE.finditer(text):
        name = match.group("name").strip()
        rest = match.group("rest")
        raw_line = match.group(0).strip()

        strength = STRENGTH_RE.search(rest)
        dose = DOSE_SLOT_RE.search(rest)
        frequency = FREQUENCY_RE.search(rest)
        duration = DURATION_RE.search(rest)

        # `value`/`unit` can only hold the drug's strength (one number) — the
        # schema has no dedicated dose-schedule fields, so the 1-0-1 / OD /
        # duration facts are expanded into plain language and folded into
        # `text`, which still starts with the line exactly as it appeared.
        phrase = ""
        if dose is not None:
            schedule = parse_dose_notation(dose)
            phrase = _dose_schedule_text(schedule)
        elif frequency is not None:
            phrase = _frequency_text(frequency.group("freq"))

        if duration is not None:
            duration_phrase = f"for {duration.group('count')} {duration.group('unit')}"
            phrase = f"{phrase}, {duration_phrase}" if phrase else duration_phrase.capitalize()

        annotated_text = f"{raw_line} [{phrase}]" if phrase else raw_line

        entities.append(
            ExtractedEntity(
                kind="medication",
                text=annotated_text,
                value=float(strength.group("value")) if strength else name,
                unit=strength.group("unit") if strength else None,
                confidence=_confidence(ocr_mean, "drug_line"),
            )
        )
    return entities


# ---------------------------------------------------------------------------
# Lab rows
# ---------------------------------------------------------------------------


def _extract_lab_rows(text: str, ocr_mean: float) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    for match in LAB_ROW_RE.finditer(text):
        analyte = match.group("analyte").strip()
        operator = match.group("operator") or ""
        raw_value = match.group("value")
        entities.append(
            ExtractedEntity(
                kind="lab",
                text=analyte,
                value=f"{operator}{raw_value}" if operator else float(raw_value),
                unit=match.group("unit"),
                reference_range=(
                    match.group("range").replace(" ", "") if match.group("range") else None
                ),
                confidence=_confidence(ocr_mean, "lab_row"),
            )
        )
    return entities


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------


def _extract_vitals(text: str, ocr_mean: float) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []

    for match in BP_RE.finditer(text):
        systolic, diastolic = match.group("systolic"), match.group("diastolic")
        entities.append(
            ExtractedEntity(
                kind="vital",
                text=f"{match.group(0).strip()} (systolic={systolic}, diastolic={diastolic})",
                value=f"{systolic}/{diastolic}",
                unit=(match.group("unit") or "mmHg").strip(),
                confidence=_confidence(ocr_mean, "bp"),
            )
        )

    _simple_vital(entities, PULSE_RE, "pulse", "bpm", text, ocr_mean)
    _simple_vital(entities, SPO2_RE, "spo2", "%", text, ocr_mean)
    _simple_vital(entities, TEMPERATURE_RE, "temperature", "F", text, ocr_mean)
    _simple_vital(entities, WEIGHT_RE, "weight", "kg", text, ocr_mean)
    _simple_vital(entities, HEIGHT_RE, "height", "cm", text, ocr_mean)
    _simple_vital(entities, RESP_RATE_RE, "resp_rate", "/min", text, ocr_mean)

    return entities


def _simple_vital(
    entities: list[ExtractedEntity],
    pattern: re.Pattern[str],
    rule_key: str,
    default_unit: str,
    text: str,
    ocr_mean: float,
) -> None:
    """Append entities for a single-value vital (pulse, SpO2, temp, weight, height, RR).

    Factored out because these six patterns are identical in shape — one
    labelled numeric value, an optional unit — and only differ in which regex
    and default unit apply. BP is handled separately since it is two numbers.
    """
    for match in pattern.finditer(text):
        unit = match.groupdict().get("unit")
        entities.append(
            ExtractedEntity(
                kind="vital",
                text=match.group(0).strip(),
                value=float(match.group("value")),
                unit=(unit or default_unit).strip(),
                confidence=_confidence(ocr_mean, rule_key),
            )
        )
