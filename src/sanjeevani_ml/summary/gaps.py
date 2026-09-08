"""A plain-language account of what the paperwork does and doesn't tell us.

Not a physician-facing summary section — this is groundwork for one. The full
Module C summary needs an honest "gaps" list (per the platform's rule that a
summary hiding what it doesn't know is worse than a short one), and the document
side of that gap analysis doesn't need Mahek's transcript to compute: it only
needs to look at what `DocumentBundle` itself already knows it couldn't do.

Every line here traces to something already sitting in the bundle — an
`unreadable` document, an entity below a confidence floor, a `warnings` entry
from timeline building or abnormal-value flagging. This module invents nothing
new; it just organises what other modules already reported into one place a
person can read in five seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sanjeevani_ml.schemas import DocumentBundle

#: A lab or medication entity below this confidence is worth calling out by name
#: as "uncertain" rather than silently trusted. Distinct from
#: `summary.document_sections._LOW_CONFIDENCE` (which hedges a sentence) — this
#: one decides whether an item is listed as a standalone concern at all, so it is
#: deliberately stricter.
_UNCERTAIN_CONFIDENCE = 0.5


@dataclass
class DocumentationGapsReport:
    """What a physician should know about the LIMITS of this document set.

    Kept as a plain dataclass rather than a schema class: this is an internal
    tool for whoever assembles the final HistorySummary, not itself a FHIR- or
    API-facing shape.
    """

    documents_processed: int
    documents_unreadable: list[str] = field(default_factory=list)
    uncertain_entities: list[str] = field(default_factory=list)
    unresolved_warnings: list[str] = field(default_factory=list)
    #: True when nothing at all could be extracted from a single provided document —
    #: the summary generator should treat this very differently from "some gaps".
    entirely_undocumented: bool = False

    def as_plain_text(self) -> str:
        """A short paragraph a physician actually has time to read."""
        if self.entirely_undocumented:
            return "No usable clinical information could be extracted from the documents provided."

        lines: list[str] = [
            f"{self.documents_processed} document(s) processed."
        ]
        if self.documents_unreadable:
            lines.append(
                f"{len(self.documents_unreadable)} could not be read at all "
                f"(patient may wish to re-photograph these)."
            )
        if self.uncertain_entities:
            preview = ", ".join(self.uncertain_entities[:3])
            more = f" and {len(self.uncertain_entities) - 3} more" if len(self.uncertain_entities) > 3 else ""
            lines.append(f"Low-confidence readings: {preview}{more}.")
        if self.unresolved_warnings:
            lines.append(f"{len(self.unresolved_warnings)} other item(s) need manual review.")
        return " ".join(lines)


def build_documentation_gaps_report(bundle: DocumentBundle) -> DocumentationGapsReport:
    """Summarise everything the document pipeline already flagged as incomplete.

    Never re-judges anything — an item only appears here because some earlier
    stage (extraction, terminology coding, abnormal-value flagging) already put
    a low number or a warning string on it. This function's only job is to make
    those already-honest signals visible in one place.
    """
    uncertain: list[str] = []
    for entry in bundle.timeline:
        for entity in entry.entities:
            if entity.kind in ("lab", "medication") and entity.confidence < _UNCERTAIN_CONFIDENCE:
                uncertain.append(entity.text)

    entirely_undocumented = (
        bundle.documents_processed > 0
        and not bundle.timeline
        and len(bundle.unreadable) == bundle.documents_processed
    )

    return DocumentationGapsReport(
        documents_processed=bundle.documents_processed,
        documents_unreadable=list(bundle.unreadable),
        uncertain_entities=uncertain,
        unresolved_warnings=list(bundle.warnings),
        entirely_undocumented=entirely_undocumented,
    )
