"""Regex vocabulary for Indian clinical documents.

This module is deliberately *only* patterns plus the small parsers that turn a
regex match into a normalised Python value. It has no dependency on the rest of
the pipeline, so every pattern here can be unit-tested in isolation and read
aloud to a judge in one sentence.

Design rules followed throughout:

* Every pattern is written verbose (``re.X``) and commented with *why* it looks
  the way it does. A clever unmaintainable regex is a liability at 2 AM.
* Nothing here maps to a terminology code and nothing here decides clinical
  meaning. Extraction only. Coding happens in ``terminology/``.
* Ambiguity is reported, never resolved silently (see :class:`ParsedDate`).
* Confidence multipliers live next to the rule that earns them, so the
  structurer can multiply OCR confidence by a rule certainty without inventing
  numbers of its own.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Final, Optional

__all__ = [
    "normalize_text",
    "ParsedDate",
    "DoseSchedule",
    "parse_date",
    "parse_dose_notation",
    "DATE_DMY_RE",
    "DOSE_SLOT_RE",
    "FREQUENCY_RE",
    "DURATION_RE",
    "STRENGTH_RE",
    "DRUG_LINE_RE",
    "BP_RE",
    "PULSE_RE",
    "SPO2_RE",
    "TEMPERATURE_RE",
    "WEIGHT_RE",
    "HEIGHT_RE",
    "RESP_RATE_RE",
    "LAB_ROW_RE",
    "REFERENCE_RANGE_RE",
    "FACILITY_RE",
    "DIAGNOSIS_LINE_RE",
    "split_conditions",
    "RULE_CONFIDENCE",
]


# ---------------------------------------------------------------------------
# Rule certainty multipliers
# ---------------------------------------------------------------------------
# These are multiplied against Mahek's OCR mean_confidence. They express "how
# sure is the *rule*, assuming the characters were read correctly" and are
# deliberately conservative: a wrong dose reported at 0.98 is dangerous, the
# same dose at 0.55 gets checked by a human.
#
# Structural patterns (a BP reading is unmistakably shaped like 120/80) score
# high. Patterns that depend on layout or free text score lower.
RULE_CONFIDENCE: Final[dict[str, float]] = {
    "bp": 0.95,
    "spo2": 0.95,
    "pulse": 0.90,
    "temperature": 0.90,
    "weight": 0.90,
    "height": 0.90,
    "resp_rate": 0.85,
    "dose_notation": 0.95,  # 1-0-1 is unambiguous when it matches
    "frequency_abbrev": 0.85,  # OD/BD/TDS are standard but OCR-fragile
    "strength": 0.90,
    "drug_line": 0.80,  # depends on a dosage-form prefix being present
    "lab_row": 0.75,  # column alignment is the weakest signal we rely on
    "date_unambiguous": 0.95,  # day > 12, so DD/MM is proven by the value
    "date_assumed_dmy": 0.70,  # both fields <= 12; we assumed Indian order
    "facility": 0.60,
    "diagnosis_line": 0.70,  # a labelled line, but free text — not shape-verified
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Devanagari digits appear on government / PHC forms and in Hindi prescriptions
# written by hand. We fold them to ASCII before matching so that a single set of
# numeric patterns serves both scripts.
_DEVANAGARI_DIGITS: Final[dict[int, str]] = {
    ord(dev): str(i) for i, dev in enumerate("०१२३४५६७८९")
}

# OCR routinely emits typographic dashes where the document had a hyphen. The
# 1-0-1 pattern lives or dies on this, so we fold dash variants to "-".
_DASH_VARIANTS: Final[dict[int, str]] = {
    ord(ch): "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
}

_WHITESPACE_RUN_RE: Final[re.Pattern[str]] = re.compile(r"[ \t\u00a0]+")


def normalize_text(text: str) -> str:
    """Fold a raw OCR string into the character space the patterns expect.

    Applies Unicode NFKC, converts Devanagari digits to ASCII digits, folds
    dash-like characters to ``-``, and collapses runs of spaces/tabs. Line
    structure is preserved because column-aligned lab printouts carry meaning in
    their line breaks.

    Note: intra-line runs of spaces are collapsed to a single space, so callers
    that need the original column offsets should keep the raw text too.

    Args:
        text: Raw text as produced by OCR.

    Returns:
        The normalised text. Never logged — it is PHI.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(_DEVANAGARI_DIGITS).translate(_DASH_VARIANTS)
    return "\n".join(_WHITESPACE_RUN_RE.sub(" ", line).strip() for line in folded.splitlines())


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTH_NAMES: Final[dict[str, int]] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

DATE_DMY_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![\d/])                      # not already inside a longer number
    (?P<first>\d{1,2})              # day in Indian order; could be month if <= 12
    [/\-.\s]
    (?P<second>\d{1,2}|[A-Za-z]{3,4}\.?)   # month, numeric or short name
    [/\-.\s]
    (?P<year>\d{4}|\d{2})           # 2026 or 26
    (?![\d/])
    """,
    re.X,
)


@dataclass(frozen=True)
class ParsedDate:
    """A date pulled out of a document, with its ambiguity made explicit.

    ``03/04/2026`` is 3 April 2026 in India and 4 March 2026 in the US. Reading
    it the Western way is a clinical error, not a formatting nitpick, so we
    always parse day-first *and* flag when the value itself could not prove the
    order. Downstream may surface that flag to the patient for confirmation.
    """

    value: date
    raw: str
    ambiguous: bool
    """True when both fields are <= 12, i.e. the DD/MM reading is an assumption."""

    @property
    def rule_confidence(self) -> float:
        """Rule certainty for this parse, to be multiplied by OCR confidence."""
        return RULE_CONFIDENCE["date_assumed_dmy" if self.ambiguous else "date_unambiguous"]


def parse_date(match: re.Match[str], *, century_pivot: int = 70) -> Optional[ParsedDate]:
    """Turn a :data:`DATE_DMY_RE` match into a :class:`ParsedDate`.

    Day-first is assumed because every Indian OPD slip, lab header and pharmacy
    register writes DD/MM/YYYY.

    Args:
        match: A match produced by :data:`DATE_DMY_RE`.
        century_pivot: Two-digit years above this map to 19xx, else 20xx.

    Returns:
        The parsed date, or ``None`` if the fields do not form a real calendar
        date (e.g. 31/02). An impossible date is dropped rather than corrected —
        we never repair a value we cannot verify.
    """
    raw = match.group(0)
    day_str, month_str, year_str = match.group("first", "second", "year")

    if month_str[0].isalpha():
        month = _MONTH_NAMES.get(month_str.rstrip(".").lower()[:4].rstrip("."))
        if month is None:
            month = _MONTH_NAMES.get(month_str.rstrip(".").lower()[:3])
        if month is None:
            return None
        ambiguous = False  # a named month proves which field is the month
    else:
        month = int(month_str)
        ambiguous = int(day_str) <= 12  # could equally have been MM/DD

    day = int(day_str)
    year = int(year_str)
    if len(year_str) == 2:
        year += 1900 if year > century_pivot else 2000

    try:
        value = date(year, month, day)
    except ValueError:
        return None
    return ParsedDate(value=value, raw=raw, ambiguous=ambiguous)


# ---------------------------------------------------------------------------
# Dosing notation
# ---------------------------------------------------------------------------

# "1-0-1" means one tablet morning, none at noon, one at night. It is the
# dominant Indian dosing notation and near-absent from Western datasets, so it
# gets a first-class pattern rather than being treated as free text.
#
# A slot is a whole number (0, 1, 2), a half written as 1/2 or ½, or a decimal
# (0.5). Four-slot forms (1-0-1-1, adding bedtime) are common enough to accept.
# A slot value is at most two digits (0, 1, 2, rarely 10 ml). Bounding it this
# way is what stops "03-04-2026" and "022-2754-1122" from looking like a dose.
# The fraction alternative (\d/\d) must come before the bare-digit alternative:
# regex alternation takes the first branch that matches, and bare digits would
# otherwise match just the "1" in "1/2" and silently leave the "/2" behind.
_SLOT = r"(?:\d{1,2}/\d{1,2}|\d{1,2}(?:\.\d{1,2})?|[½¼¾])"

DOSE_SLOT_RE: Final[re.Pattern[str]] = re.compile(
    rf"""
    (?<![\w/.-])
    (?P<morning>{_SLOT}) \s*-\s*
    (?P<noon>{_SLOT})    \s*-\s*
    (?P<evening>{_SLOT})
    (?: \s*-\s* (?P<night>{_SLOT}) )?     # optional 4th slot = bedtime
    (?![\w.-])
    """,
    re.X,
)

_VULGAR_FRACTIONS: Final[dict[str, float]] = {"½": 0.5, "¼": 0.25, "¾": 0.75}


@dataclass(frozen=True)
class DoseSchedule:
    """Units taken per slot across the day, parsed from ``1-0-1`` notation."""

    morning: float
    noon: float
    evening: float
    night: Optional[float] = None
    raw: str = ""

    @property
    def total_per_day(self) -> float:
        """Total units per day. Used for dose sanity checks *outside* this module."""
        return self.morning + self.noon + self.evening + (self.night or 0.0)

    @property
    def slots(self) -> list[float]:
        parts = [self.morning, self.noon, self.evening]
        if self.night is not None:
            parts.append(self.night)
        return parts


def _slot_value(token: str) -> float:
    if token in _VULGAR_FRACTIONS:
        return _VULGAR_FRACTIONS[token]
    if "/" in token:
        numerator, _, denominator = token.partition("/")
        return int(numerator) / int(denominator)
    return float(token)


def parse_dose_notation(match: re.Match[str]) -> DoseSchedule:
    """Turn a :data:`DOSE_SLOT_RE` match into a :class:`DoseSchedule`.

    No clinical judgement is applied here: a schedule of 4-4-4 is parsed exactly
    as written. Whether a dose is plausible is a medical question and is not
    decided by a regex module.
    """
    night_token = match.group("night")
    return DoseSchedule(
        morning=_slot_value(match.group("morning")),
        noon=_slot_value(match.group("noon")),
        evening=_slot_value(match.group("evening")),
        night=_slot_value(night_token) if night_token else None,
        raw=match.group(0),
    )


# ---------------------------------------------------------------------------
# Prescription lines
# ---------------------------------------------------------------------------

# Latin frequency abbreviations still dominate handwritten and printed Indian
# prescriptions. Written as an alternation of exact tokens so that "ODS" or a
# word starting with "bd" cannot match.
FREQUENCY_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<freq>
        OD | BD | BID | TDS | TID | QID | QDS | HS | SOS | PRN | STAT | Q\d{1,2}H
    )
    (?![A-Za-z])
    """,
    re.X | re.I,
)

# "x 5 days", "for 7 days", "5 दिन", "1 week".
DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?: (?:x|X|for|\u0915\u0947\s*\u0932\u093f\u090f) \s* )?
    (?P<count>\d{1,3})
    \s*
    (?P<unit> days? | din | दिन | weeks? | हफ़्ते | हफ्ते | months? | माह | महीने )
    (?![A-Za-z])
    """,
    re.X | re.I,
)

# Strength as printed on the strip: 500mg, 12.5 mg, 5 ml, 40 IU, 0.5 mcg.
STRENGTH_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit> mg | mcg | µg | ug | gm | g | ml | mL | IU | U | % )
    (?![A-Za-z])
    """,
    re.X,
)

# A prescription line is recognised by its dosage-form prefix, which Indian
# prescriptions almost always carry: "Tab. Glycomet 500mg 1-0-1 x 30 days".
# We capture the *name as written* — brand resolution is terminology/'s job,
# not extraction's.
DRUG_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^\s*
    (?: \d{1,2} [.)] \s* )?                  # optional list numbering "1." / "2)"
    (?P<form>
        Tab | Tabs | Tablet | Cap | Caps | Capsule | Syp | Syrup | Susp |
        Inj | Injection | Oint | Ointment | Drops? | Gel | Cream | Powder |
        गोली | कैप्सूल | सिरप | इंजेक्शन
    )
    \.?\s+
    (?P<name> [A-Za-z\u0900-\u097F][A-Za-z0-9\u0900-\u097F'\- ]{1,48}? )
    (?=\s*(?:\d|$))                          # name ends where strength/dose starts
    (?P<rest> .* )$
    """,
    re.X | re.M | re.I,
)


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------
# Each vital accepts its English label, common abbreviations, and the Hindi
# label seen on PHC and government OPD slips. An English-only parser fails on
# exactly the records this project exists to serve.

BP_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> BP | B\.P\.? | Blood \s* Pressure | बीपी | बी\.पी\.? | रक्तचाप )
    \s* [:\-]? \s*
    (?P<systolic>\d{2,3}) \s* / \s* (?P<diastolic>\d{2,3})
    \s* (?P<unit> mmHg | mm \s* Hg )?
    """,
    re.X | re.I,
)

PULSE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> Pulse | PR | Heart \s* Rate | HR | नाड़ी | नाडी )
    \s* [:\-]? \s*
    (?P<value>\d{2,3})
    \s* (?P<unit> bpm | /min | per \s* min )?
    """,
    re.X | re.I,
)

SPO2_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> SpO2 | SPO2 | SaO2 | O2 \s* Sat | Oxygen \s* Saturation | ऑक्सीजन )
    \s* [:\-]? \s*
    (?P<value>\d{2,3}) \s* (?P<unit>%)?
    """,
    re.X | re.I,
)

TEMPERATURE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> Temp | Temperature | तापमान | बुखार )
    \.? \s* [:\-]? \s*
    (?P<value>\d{2,3}(?:\.\d)?)
    \s* (?: ° | deg \s* )? \s* (?P<unit> F | C )?
    (?![A-Za-z])
    """,
    re.X | re.I,
)

WEIGHT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> Weight | Wt | वजन | भार )
    \.? \s* [:\-]? \s*
    (?P<value>\d{1,3}(?:\.\d{1,2})?)
    \s* (?P<unit> kgs? | kilograms? | किलो )
    """,
    re.X | re.I,
)

HEIGHT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> Height | Ht | लंबाई | कद )
    \.? \s* [:\-]? \s*
    (?P<value>\d{2,3}(?:\.\d)?)
    \s* (?P<unit> cms? | centimet(?:er|re)s? | m )
    (?![A-Za-z])
    """,
    re.X | re.I,
)

RESP_RATE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<label> RR | Resp \s* Rate | Respiratory \s* Rate | श्वसन )
    \.? \s* [:\-]? \s*
    (?P<value>\d{1,2})
    \s* (?P<unit> /min | per \s* min | bpm )?
    """,
    re.X | re.I,
)


# ---------------------------------------------------------------------------
# Lab report rows
# ---------------------------------------------------------------------------

# Indian lab printouts are column-aligned text, not delimited data:
#
#   Haemoglobin            11.2      g/dL        13.0 - 17.0
#   Fasting Blood Sugar    126       mg/dL       70 - 100
#   HbA1c                  7.8       %
#
# The row is recognised by shape: an analyte name, then a numeric result, then
# an optional unit, then an optional reference range. Units are captured as
# written; unit chaos (mg/dL vs mmol/L) is normalised in terminology/, because
# converting a value is a clinical decision and belongs next to the code that
# knows which analyte it is.
REFERENCE_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<low>\d+(?:\.\d+)?) \s* (?: - | to | – ) \s* (?P<high>\d+(?:\.\d+)?)
    """,
    re.X | re.I,
)

_UNIT = r"""(?:
        mg/dL | mg/dl | g/dL | g/dl | mmol/L | mmol/l | mEq/L | µmol/L | umol/L |
        IU/L | U/L | mIU/L | ng/mL | ng/ml | pg/mL | µg/dL | ug/dL |
        cells/cumm | /cumm | lakhs?/cumm | million/cumm | 10\^\d/L |
        fL | pg | % | mm/hr | mL/min
)"""

LAB_ROW_RE: Final[re.Pattern[str]] = re.compile(
    rf"""
    ^\s*
    (?P<analyte> [A-Za-z\u0900-\u097F][A-Za-z0-9\u0900-\u097F()\-.,/ ]{{1,44}}? )
    \s* [:.]? \s{{1,}}                       # column gap (or a colon on a dense slip)
    (?P<operator> [<>] )? \s*
    (?P<value> \d+(?:\.\d+)? )
    (?: \s* (?P<unit> {_UNIT}) )?
    (?: [\s|]* \(? (?P<range> \d+(?:\.\d+)? \s* (?:-|to|–) \s* \d+(?:\.\d+)? ) \)? )?
    \s*$
    """,
    re.X | re.M,
)


# ---------------------------------------------------------------------------
# Facility
# ---------------------------------------------------------------------------
# Header lines naming the issuing facility. Low confidence by design: this is a
# heuristic on free text and is only ever used to populate a *draft*
# performer/organisation reference for the patient to confirm.
FACILITY_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^ .{0,80}?
    (?P<facility>
        # The distinguishing name ("Sunrise", "Koparkhairane") is optional: many
        # government headers are just "PRIMARY HEALTH CENTRE, <place>".
        (?: [A-Za-z\u0900-\u097F][A-Za-z0-9\u0900-\u097F&'.,\- ]{2,60}? \s* )?
        (?: Primary \s+ Health \s+ Centre | PHC | CHC | Sub \s* -? \s* Centre |
            District \s+ Hospital | Civil \s+ Hospital | Medical \s+ College |
            Hospital | Nursing \s+ Home | Clinic | Polyclinic |
            Diagnostics? | Diagnostic \s+ Cent(?:re|er) | Laboratory | Labs? |
            Path \s* Lab | अस्पताल | चिकित्सालय | स्वास्थ्य \s* केंद्र )
    )
    """,
    re.X | re.I | re.M,
)


# ---------------------------------------------------------------------------
# Diagnosis lines
# ---------------------------------------------------------------------------
# A labelled line ("Dx:", "Diagnosis:", "Impression:", "निदान:") followed by one
# or more conditions, often comma-separated. We only recognise the *label* by
# shape; the condition text itself is free text and is not validated here —
# that is terminology/'s job (SNOMED/ICD-11 lookup, never-invent-a-code).
DIAGNOSIS_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    ^\s*
    (?: Dx | Diagnosis | Impression | निदान )
    \s* [.:]\s*
    (?P<conditions> .+ )
    $
    """,
    re.X | re.M | re.I,
)

# Splits a matched `conditions` group into individual condition phrases, so that
# "Type 2 Diabetes Mellitus, Hypertension" becomes two entities — each needs its
# own terminology code downstream, so one entity per condition is the useful unit.
_CONDITION_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"\s*(?:,|;|\band\b)\s*", re.I)


def split_conditions(conditions_text: str) -> list[str]:
    """Split a diagnosis line's free text into individual condition phrases.

    Args:
        conditions_text: The ``conditions`` group from a :data:`DIAGNOSIS_LINE_RE`
            match, e.g. ``"Type 2 Diabetes Mellitus, Hypertension"``.

    Returns:
        Non-empty, stripped condition phrases in their original order.
    """
    return [part.strip() for part in _CONDITION_SPLIT_RE.split(conditions_text) if part.strip()]