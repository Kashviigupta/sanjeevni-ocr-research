"""Put the documents in the order a physician reads a history in.

Oldest first — a medical history reads forward, and the physician's eye lands on the
most recent entry at the bottom next to the current complaint.

Undated documents go into a trailing block, never interleaved. We do NOT infer a date
from upload order or file order: that is a guess wearing a timestamp, and once it is
rendered next to real dates nobody can tell the difference.
"""
from __future__ import annotations

from sanjeevani_ml.schemas import TimelineEntry

#: Tie-break for documents sharing a date. Investigations precede the prescription
#: written off them, and a discharge/note frames both. Ordering only — it asserts
#: nothing clinical.
TYPE_PRIORITY: dict[str, int] = {
    "note": 0,        # discharge summaries, clinical notes — the framing document
    "condition": 1,
    "imaging": 2,
    "lab": 3,
    "allergy": 4,
    "prescription": 5,  # written after the results it responds to
}


def sort_key(entry: TimelineEntry) -> tuple:
    """Dated ascending, then type priority, then title for a stable order."""
    return (
        entry.date or "",
        TYPE_PRIORITY.get(entry.kind, 99),
        entry.title.lower(),
    )


def order(entries: list[TimelineEntry]) -> list[TimelineEntry]:
    """Dated documents oldest-first, then every undated document."""
    dated = sorted((e for e in entries if e.date), key=sort_key)
    undated = sorted(
        (e for e in entries if not e.date),
        key=lambda e: (TYPE_PRIORITY.get(e.kind, 99), e.title.lower()),
    )
    return [*dated, *undated]
