"""Collapse the same document handed to us more than once.

Patients bring the original, the pharmacy's carbon copy, and a phone photo of the same
page. Showing a physician the same prescription three times wastes the two minutes we
exist to protect.

The merge here is LOSSLESS: every contributing scan stays openable via
``duplicate_of``, and entities are unioned rather than replaced. That is what makes a
merge-friendly threshold defensible — a false merge costs a line in the timeline, not a
document.

Two guards override the threshold, both clinical:

* **Dose disagreement blocks a merge.** Glycomet 500 and Glycomet 1000 on the same day
  may be a dose change, not a duplicate scan. Surface both.
* **Any date difference blocks a merge.** Two identical labs a day apart may be
  collection vs report date, or may be two draws. We do not decide which.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sanjeevani_ml.schemas import ExtractedEntity, TimelineEntry
from sanjeevani_ml.terminology.mapper import generic_name

#: Jaccard overlap of normalised entity keys above which two same-type, same-date
#: documents are considered the same document.
#:
#: 0.6, not 0.8: Kashvi's call. A false split shows the physician the same prescription
#: twice, which is the failure mode we are trying to remove; a false merge only shortens
#: the timeline, because both documents remain reachable. If merging ever becomes
#: destructive, RAISE THIS NUMBER FIRST.
DUPLICATE_THRESHOLD = 0.6

#: Below this many shared keys, Jaccard is noise. Two documents with one entity each
#: that happen to match are not evidence of anything.
_MIN_KEYS_FOR_COMPARISON = 2

_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b", re.IGNORECASE)

#: Dosage-form and frequency words that surround a drug name on a real prescription
#: line ("Tab. Glycomet 500mg 1-0-1 x 30 days"). Filtering these out, along with all
#: digits, is what lets us find the name itself instead of feeding a whole line to
#: `generic_name`, which only knows single drug names.
_DRUG_LINE_STOPWORDS = {
    "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules",
    "syp", "syrup", "inj", "injection", "oint", "ointment", "drops", "susp",
    "suspension", "x", "days", "day", "od", "bd", "tds", "qid", "sos", "mg",
    "mcg", "ml", "iu",
}


def drug_name_tokens(text: str) -> list[str]:
    """Alphabetic words in a medication line, with dosage-form noise removed.

    Doses and day-counts are digits; frequency codes and forms are the stopwords
    above. What's left is the drug name — one or two words, brand or generic.
    """
    words = re.findall(r"[A-Za-z]+", text)
    return [w for w in words if w.lower().rstrip(".") not in _DRUG_LINE_STOPWORDS]


def primary_drug_name(text: str) -> str | None:
    """Best-guess single drug name out of a medication line, brand or generic.

    Tries every candidate word against the terminology table first — that is the
    word a prescriber actually meant. Falls back to the first alphabetic word so an
    unrecognised drug still gets *a* name rather than none.
    """
    candidates = drug_name_tokens(text)
    for word in candidates:
        if generic_name(word):
            return word
    return candidates[0] if candidates else None


def entity_key(entity: ExtractedEntity) -> str:
    """A comparable identity for one extracted fact, robust to OCR wobble.

    Medications key on the drug name alone (brand or generic), never on the dose —
    a dose difference is a clinical signal handled separately by `_dose_conflict`,
    not evidence that two documents are unrelated. Labs and diagnoses normalise on
    lowercased text with punctuation stripped.
    """
    if entity.kind == "medication":
        name = primary_drug_name(entity.text)
        base = (generic_name(name) if name else None) or name or entity.text
        base = base.lower()
    else:
        base = entity.text.lower()
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", "", base)


def _dose_of(entity: ExtractedEntity) -> str | None:
    """The strength written on a medication line, normalised, or None."""
    match = _DOSE_RE.search(entity.text)
    if not match:
        return None
    return f"{float(match.group(1)):g}{match.group(2).lower().rstrip('s')}"


def _keys(entry: TimelineEntry) -> set[str]:
    return {k for k in (entity_key(e) for e in entry.entities) if k}


def jaccard(left: set[str], right: set[str]) -> float:
    """Overlap of two key sets. 0.0 when either side is empty."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dose_conflict(a: TimelineEntry, b: TimelineEntry) -> bool:
    """True when the same generic drug carries different strengths across the two.

    This is a clinical guard, not an optimisation: a dose change is exactly the kind of
    thing a physician must see, and exactly what an eager de-duplicator would erase.
    """
    def doses(entry: TimelineEntry) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for entity in entry.entities:
            if entity.kind != "medication":
                continue
            dose = _dose_of(entity)
            if dose:
                out.setdefault(entity_key(entity), set()).add(dose)
        return out

    left, right = doses(a), doses(b)
    return any(left[key] != right[key] for key in left.keys() & right.keys())


def is_duplicate(a: TimelineEntry, b: TimelineEntry) -> tuple[bool, str]:
    """Are these two entries the same underlying document?

    Returns ``(verdict, reason)``; the reason is recorded so a judge — or a physician
    wondering where their second prescription went — can be told exactly why.
    """
    if a.kind != b.kind:
        return False, "different document types"

    # Two undated documents are never merged with each other. Without a date to
    # anchor on, "same drugs" is not enough evidence they are the same physical
    # page — a repeat prescription looks identical to a re-photographed one.
    if a.date is None and b.date is None:
        return False, "both undated — cannot confirm duplicate without a date"

    # Any date difference blocks a merge, including dated-vs-undated. Kashvi's call:
    # collection date vs report date is indistinguishable from two separate draws, and
    # we do not guess between them.
    if a.date != b.date:
        return False, "different (or missing) clinical dates"

    keys_a, keys_b = _keys(a), _keys(b)
    if min(len(keys_a), len(keys_b)) < _MIN_KEYS_FOR_COMPARISON:
        return False, "too few extracted entities to compare meaningfully"

    if _dose_conflict(a, b):
        return False, "same drug at different strengths — possible dose change, kept separate"

    score = jaccard(keys_a, keys_b)
    if score >= DUPLICATE_THRESHOLD:
        return True, f"entity overlap {score:.2f} >= {DUPLICATE_THRESHOLD}"
    return False, f"entity overlap {score:.2f} < {DUPLICATE_THRESHOLD}"


@dataclass
class MergeNote:
    """One de-duplication decision, for the warnings list and for debugging."""

    kept: str | None
    absorbed: str | None
    reason: str


def merge(primary: TimelineEntry, other: TimelineEntry) -> TimelineEntry:
    """Fold ``other`` into ``primary`` without losing anything.

    The higher-confidence scan wins the headline fields; entities are unioned by key so
    a value only one copy captured survives; both document ids remain reachable.
    """
    merged_entities = list(primary.entities)
    seen = _keys(primary)
    for entity in other.entities:
        key = entity_key(entity)
        if key and key not in seen:
            merged_entities.append(entity)
            seen.add(key)

    duplicate_of = list(dict.fromkeys(
        [*primary.duplicate_of, *other.duplicate_of, *( [other.document_id] if other.document_id else [] )]
    ))

    return primary.model_copy(update={
        "entities": merged_entities,
        "facility": primary.facility or other.facility,
        "duplicate_of": [d for d in duplicate_of if d != primary.document_id],
    })


def deduplicate(entries: list[TimelineEntry]) -> tuple[list[TimelineEntry], list[MergeNote]]:
    """Collapse duplicates across a whole upload set.

    The higher-confidence copy of a pair becomes the primary, so the physician reads the
    best scan we got.
    """
    kept: list[TimelineEntry] = []
    notes: list[MergeNote] = []

    for entry in entries:
        for index, existing in enumerate(kept):
            duplicate, reason = is_duplicate(existing, entry)
            if not duplicate:
                continue
            primary, absorbed = existing, entry
            if _entry_confidence(entry) > _entry_confidence(existing):
                primary, absorbed = entry, existing
            kept[index] = merge(primary, absorbed)
            notes.append(MergeNote(
                kept=primary.document_id, absorbed=absorbed.document_id, reason=reason,
            ))
            break
        else:
            kept.append(entry)

    return kept, notes


def _entry_confidence(entry: TimelineEntry) -> float:
    """Mean entity confidence — how well we read this particular copy."""
    if not entry.entities:
        return 0.0
    return sum(e.confidence for e in entry.entities) / len(entry.entities)
