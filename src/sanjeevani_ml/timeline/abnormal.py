"""Flag out-of-range lab values for physician attention.

Decision rule (confirmed with the team): the document's own printed reference
range is authoritative — the lab that ran the test is the authority on its own
numbers, because ranges are method- and population-specific. The curated
terminology table is a FALLBACK only, used when the document prints no range at
all. A genuine disagreement between the two is never resolved silently; it goes
into `DocumentBundle.warnings` so a physician can see the lab's range differs
from the textbook one, without us picking a side.

Two guards sit in front of that rule:

* A UNIT mismatch (mg/dL vs mmol/L is the common one) means the two ranges are
  not comparable at all — we do not convert units silently, because a silent
  conversion that is wrong is worse than declining to compare.
* A document range that is not just different but NONSENSICAL — inverted, or
  implausibly narrow/wide next to the curated range — is treated as an OCR
  failure, not a genuine clinical range. We fall back to curated and warn,
  rather than judging a value "abnormal" against a range Tesseract garbled.

If neither the document nor the curated table has a usable range, the value is
reported with no `direction` at all. Reporting an out-of-range verdict with a
threshold we do not actually have would be a guess wearing a clinical opinion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sanjeevani_ml.schemas import AbnormalValue, DocumentBundle, ExtractedEntity
from sanjeevani_ml.terminology.loader import load_lab_terms
from sanjeevani_ml.terminology.mapper import normalize

#: A document's range is treated as an OCR garble — not a genuine differing
#: range — when its width falls outside [curated_width / 2, curated_width * 2].
#: Arbitrary, same status as `timeline.dedup.DUPLICATE_THRESHOLD`: it decides
#: whether a physician sees "this lab's own range" or "our textbook range plus
#: a warning that OCR may have mangled the printed one". Tune against real
#: lab printouts, not in the abstract.
_GARBLE_WIDTH_RATIO = 2.0

_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")

#: Units that are the same physical quantity spelled differently. Anything not
#: in the same group is a hard mismatch, not a conversion opportunity — we do
#: not convert, we decline to compare and say why.
_UNIT_ALIASES: dict[str, str] = {
    "mg/dl": "mg/dl", "mg/dL": "mg/dl",
    "g/dl": "g/dl", "g/dL": "g/dl",
    "%": "%", "percent": "%",
    "mmhg": "mmhg", "mmHg": "mmhg",
    "iu/l": "iu/l", "u/l": "iu/l",
    "cells/cumm": "cells/cumm", "/cumm": "cells/cumm",
}


def _canonical_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return _UNIT_ALIASES.get(unit.strip().lower(), unit.strip().lower())


@dataclass(frozen=True)
class _Range:
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        """Inclusive both ends: a value AT the boundary is normal, not abnormal.

        This is the conventional clinical reading of a printed reference range
        — "70-100" means 100 itself is still within range, not the first
        abnormal value.
        """
        return self.low <= value <= self.high


def _parse_range(text: str | None) -> _Range | None:
    """"70-100" -> Range(70, 100). None for anything that doesn't parse cleanly.

    An unparsed range is treated exactly like a missing one — we never guess
    at what a malformed string was trying to say.
    """
    if not text:
        return None
    match = _RANGE_RE.match(text)
    if not match:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    if low >= high:
        return None  # inverted or degenerate — not a usable range, garbled or not
    return _Range(low, high)


def _curated_range(entity: ExtractedEntity) -> tuple[_Range | None, str | None]:
    """The curated table's range and unit for this analyte, if the table has one.

    Looks up by terminology code first (exact — this entity has already been
    through `terminology.mapper.code_result`), falling back to a normalised
    text match against the analyte name for entities that were not coded.

    Returns `(None, None)` if the table has no range for this analyte at all —
    that is a real gap in `ml/data/lab_terms.csv`, not a bug here, and it is
    exactly what should be reported as "not implemented for X" rather than
    silently treated as "checked and normal".
    """
    target_code = entity.code.code if entity.code else None
    target_text = normalize(entity.text)

    for row in load_lab_terms():
        matches_code = target_code is not None and row.get("code") == target_code
        matches_text = normalize(row.get("term", "")) == target_text
        if not (matches_code or matches_text):
            continue
        # Column name isn't confirmed to exist yet in lab_terms.csv — try the
        # likely spellings rather than assuming one. If none are present,
        # this analyte simply has no curated range, which is a real gap to
        # flag for whoever maintains the CSV, not a code path to fake.
        range_text = row.get("reference_range") or row.get("range")
        if not range_text and row.get("range_low") and row.get("range_high"):
            range_text = f"{row['range_low']}-{row['range_high']}"
        parsed = _parse_range(range_text)
        unit = row.get("unit") or None
        return parsed, unit
    return None, None


def _is_garbled(document_range: _Range, curated_range: _Range) -> bool:
    """Does the printed range look like an OCR failure rather than a real one?

    A genuinely different clinical range (a different assay's normal band) is
    still a plausible width. A range that's ballooned or collapsed relative to
    the textbook one is much more likely a dropped or duplicated digit.
    """
    if curated_range.width <= 0:
        return False
    ratio = document_range.width / curated_range.width
    return ratio < (1 / _GARBLE_WIDTH_RATIO) or ratio > _GARBLE_WIDTH_RATIO


def _numeric_value(entity: ExtractedEntity) -> float | None:
    if isinstance(entity.value, (int, float)):
        return float(entity.value)
    return None  # a value like ">200" or a non-numeric flag is never force-parsed


def evaluate_lab_entity(
    entity: ExtractedEntity, *, document_id: str | None
) -> tuple[AbnormalValue | None, list[str]]:
    """One lab entity -> an AbnormalValue if it's out of range, plus any warnings.

    Returns `(None, warnings)` when the value is in range, or when no usable
    range exists at all — in the latter case a warning says so, because an
    absent `AbnormalValue` must never be misread as "checked and normal".
    """
    warnings: list[str] = []
    value = _numeric_value(entity)
    if value is None:
        return None, []  # non-numeric lab result (e.g. "Positive") — not this function's job

    document_range = _parse_range(entity.reference_range)
    curated_range, curated_unit = _curated_range(entity)

    chosen_range = document_range
    chosen_unit = entity.unit

    if document_range is not None and curated_range is not None:
        doc_unit, cur_unit = _canonical_unit(entity.unit), _canonical_unit(curated_unit)
        if doc_unit and cur_unit and doc_unit != cur_unit:
            # Guard 1: never compare mg/dL to mmol/L by pretending they match.
            warnings.append(
                f"{entity.text}: document unit '{entity.unit}' differs from curated "
                f"unit '{curated_unit}' — ranges not compared, abnormality not assessed"
            )
            return None, warnings
        if _is_garbled(document_range, curated_range):
            # Guard 2: the printed range looks like an OCR error, not real data.
            warnings.append(
                f"{entity.text}: printed range '{entity.reference_range}' looks "
                f"implausible next to the curated range — using curated range instead"
            )
            chosen_range, chosen_unit = curated_range, curated_unit
        elif document_range.low != curated_range.low or document_range.high != curated_range.high:
            # A real disagreement, not garbage — surfaced, not resolved.
            warnings.append(
                f"{entity.text}: document range '{entity.reference_range}' differs from "
                f"the curated range ({curated_range.low}-{curated_range.high}); "
                f"using the document's own range for this result"
            )
    elif document_range is None and curated_range is not None:
        chosen_range, chosen_unit = curated_range, curated_unit
    elif document_range is None and curated_range is None:
        warnings.append(
            f"{entity.text}: no reference range on the document or in the curated "
            f"table — value reported without an abnormal/normal judgement"
        )
        return None, warnings

    if chosen_range is None or chosen_range.contains(value):
        return None, warnings

    direction = "high" if value > chosen_range.high else "low"
    abnormal = AbnormalValue(
        analyte=entity.text,
        value=value,
        unit=chosen_unit,
        reference_range=f"{chosen_range.low}-{chosen_range.high}",
        direction=direction,
        code=entity.code,
        source_document_id=document_id,
    )
    return abnormal, warnings


def flag_abnormal_values(bundle: DocumentBundle) -> tuple[list[AbnormalValue], list[str]]:
    """Every abnormal lab result across the whole timeline, plus any warnings.

    Pure function over an already-built bundle — call this right after
    `build_timeline` and assign the results into
    `bundle.abnormal_values` / `bundle.warnings` (see `builder.py`).
    """
    abnormal_values: list[AbnormalValue] = []
    warnings: list[str] = []
    for entry in bundle.timeline:
        for entity in entry.entities:
            if entity.kind != "lab":
                continue
            result, entity_warnings = evaluate_lab_entity(entity, document_id=entry.document_id)
            warnings.extend(entity_warnings)
            if result is not None:
                abnormal_values.append(result)
    return abnormal_values, warnings
