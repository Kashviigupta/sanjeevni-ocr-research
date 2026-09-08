"""Pick THE clinical date of a document out of the several dates printed on it.

A prescription carries a visit date, a "review after" date and sometimes a printed
footer date. A lab report carries a collection date and a reporting date. Getting this
wrong silently reorders a patient's history, so the rule here is deliberately dumb and
explainable: a date sitting next to a clinical label beats a bare date, and a date we
cannot justify comes back as ``None`` rather than as a guess.

No dependency on the extraction package beyond ``extract_dates`` — this module is pure
text in, ``(iso, confidence)`` out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

#: Labels that mark the date a document is ABOUT, scored by how directly they do so.
#: Highest wins. Every entry here is justifiable out loud, which is the bar for a regex
#: in this repo.
CLINICAL_DATE_LABELS: dict[str, float] = {
    # the event itself
    "collected on": 1.00,
    "collection date": 1.00,
    "sample collected": 1.00,
    "date of admission": 1.00,
    "admitted on": 1.00,
    "date of discharge": 1.00,
    "discharged on": 1.00,
    "date of visit": 1.00,
    "visit date": 1.00,
    # generic, still labelled
    "date": 0.85,
    "dated": 0.85,
    "dt": 0.75,
    "दिनांक": 0.85,
    # a report ABOUT an earlier event; usable but second choice
    "reported on": 0.70,
    "report date": 0.70,
    "printed on": 0.45,
    "generated on": 0.45,
}

#: Dates that belong to the FUTURE of the encounter, never to its timeline position.
#: "Review after 10/05/2026" is an instruction, not a clinical date.
FORWARD_LOOKING_LABELS: tuple[str, ...] = (
    "review", "follow up", "follow-up", "f/u", "next visit", "revisit",
    "valid till", "valid up to", "expiry", "exp",
)

#: How far back a document may plausibly be. A patient can genuinely carry a 40-year-old
#: record; a year of 1899 is an OCR failure, not a medical history.
_MAX_AGE_YEARS = 60

_DATE_RE = re.compile(
    r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b"
)
#: How much text before a date we look in for a label. One short line, no more — a label
#: three lines up is not labelling this date.
_LOOKBEHIND = 40


@dataclass(frozen=True)
class DateCandidate:
    """One date found on the page, with why we scored it the way we did."""

    iso: str
    score: float
    label: str | None
    reason: str


def _to_iso(day: int, month: int, year: int, *, today: date) -> tuple[str | None, float]:
    """Build an ISO date from DD, MM, YY(YY) parts.

    India writes DD/MM/YYYY. Reading 03/04/2026 as 4 March is a clinical error, so the
    day-first reading is the ONLY reading — we never swap on the grounds that it would
    make a valid date. An impossible day-first date is an unusable date.

    Returns ``(iso, penalty_multiplier)``; the multiplier discounts a 2-digit year,
    which we had to complete ourselves.
    """
    penalty = 1.0
    if year < 100:
        # Two-digit year. Assume this century unless that puts the document in the
        # future, in which case it is the last one. We are less sure either way.
        year = 2000 + year
        if year > today.year:
            year -= 100
        penalty = 0.75

    try:
        parsed = date(year, month, day)
    except ValueError:
        return None, 0.0

    if parsed > today:
        return None, 0.0
    if today.year - parsed.year > _MAX_AGE_YEARS:
        return None, 0.0
    return parsed.isoformat(), penalty


def find_date_candidates(text: str, *, today: date | None = None) -> list[DateCandidate]:
    """Every plausible clinical date on the page, best first.

    Forward-looking dates ("review after ...") are excluded entirely rather than
    down-scored — they are never the date of the document.
    """
    today = today or date.today()
    candidates: list[DateCandidate] = []

    for match in _DATE_RE.finditer(text):
        day, month, year = (int(g) for g in match.groups())
        iso, penalty = _to_iso(day, month, year, today=today)
        if iso is None:
            continue

        before = text[max(0, match.start() - _LOOKBEHIND):match.start()].lower()

        if any(tok in before for tok in FORWARD_LOOKING_LABELS):
            continue  # an instruction to the patient, not a date to sort on

        label, weight = None, 0.35  # a bare, unlabelled date is weak but not worthless
        for candidate_label, candidate_weight in CLINICAL_DATE_LABELS.items():
            if candidate_label in before and candidate_weight > weight:
                label, weight = candidate_label, candidate_weight

        reason = f"labelled '{label}'" if label else "unlabelled date on page"
        candidates.append(
            DateCandidate(iso=iso, score=weight * penalty, label=label, reason=reason)
        )

    candidates.sort(key=lambda c: (-c.score, c.iso))
    return candidates


def resolve_clinical_date(
    text: str,
    *,
    ocr_confidence: float = 1.0,
    today: date | None = None,
) -> tuple[str | None, float, str]:
    """The one date this document sorts on.

    Returns ``(iso_or_none, confidence, reason)``. Confidence is the label score scaled
    by OCR confidence — it can never exceed how well we read the page, because a date
    we misread is not made truer by sitting next to a good label.
    """
    candidates = find_date_candidates(text, today=today)
    if not candidates:
        return None, 0.0, "no usable date found on the document"

    best = candidates[0]
    confidence = best.score * max(0.0, min(1.0, ocr_confidence))

    # Two equally-well-labelled but DIFFERENT dates: we cannot justify a pick, so we
    # keep the earlier one and say so with reduced confidence. Surfacing the doubt is
    # the point; the physician can open the scan.
    tied = [c for c in candidates if abs(c.score - best.score) < 1e-9]
    if len({c.iso for c in tied}) > 1:
        best = min(tied, key=lambda c: c.iso)
        confidence *= 0.6
        return (
            best.iso,
            confidence,
            f"ambiguous: {len({c.iso for c in tied})} equally-labelled dates, earliest kept",
        )

    return best.iso, confidence, best.reason
