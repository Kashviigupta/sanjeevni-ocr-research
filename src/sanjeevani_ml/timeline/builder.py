"""ExtractionResult[] -> one coherent, de-duplicated, date-ordered DocumentBundle.

This is the module the PS calls out by name ("chronological organisation") and the thing
that separates us from a scanner. Pure function: no I/O, no network, no clock except the
one you pass in. ``POST /documents/timeline`` is a thin wrapper over ``build_timeline``.
"""
from __future__ import annotations

from datetime import date

from sanjeevani_ml.interactions.checker import check
from sanjeevani_ml.schemas import DocumentBundle, ExtractionResult, TimelineEntry
from sanjeevani_ml.timeline.abnormal import flag_abnormal_values
from sanjeevani_ml.timeline.dating import resolve_clinical_date
from sanjeevani_ml.timeline.dedup import deduplicate, primary_drug_name
from sanjeevani_ml.timeline.ordering import order

#: Below this, we could not read the page well enough to put anything from it in front
#: of a physician. It goes to `unreadable` and the patient is offered a retake — never
#: silently dropped.
UNREADABLE_CONFIDENCE = 0.30


def _to_entry(
    result: ExtractionResult,
    document_id: str,
    *,
    today: date | None,
    source_text: str | None,
) -> tuple[TimelineEntry, list[str]]:
    """One extraction result as a timeline entry, re-dating it if we can do better."""
    warnings: list[str] = []

    clinical_date = result.clinical_date
    date_confidence = 0.0
    if source_text:
        # Re-resolve against the full page: the structurer takes the first labelled
        # date, we weigh every date on the page against its label.
        resolved, date_confidence, reason = resolve_clinical_date(
            source_text, ocr_confidence=result.mean_confidence, today=today
        )
        if resolved:
            clinical_date = resolved
        if "ambiguous" in reason:
            warnings.append(f"{document_id}: {reason}")
    elif clinical_date:
        # Trusted from extraction, but we did not verify it ourselves. Say so rather
        # than borrowing the page's OCR confidence for a field we did not check.
        date_confidence = min(result.mean_confidence, 0.6)

    if clinical_date is None:
        warnings.append(f"{document_id}: no clinical date — placed in the undated block")

    entry = TimelineEntry(
        date=clinical_date,
        date_confidence=date_confidence,
        kind=result.suggested_type,
        title=result.suggested_title,
        facility=result.facility,
        document_id=document_id,
        entities=list(result.entities),
        fhir=result.fhir,
    )
    return entry, warnings


def build_timeline(
    results: list[ExtractionResult],
    *,
    session_id: str,
    document_ids: list[str],
    source_texts: list[str | None] | None = None,
    today: date | None = None,
) -> DocumentBundle:
    """Build the patient's document history.

    ``results[i]`` corresponds to ``document_ids[i]``; ``source_texts[i]`` is the OCR
    text of that page when available, which lets us re-weigh the dates on it.

    Raises ``ValueError`` on a length mismatch — a document silently paired with the
    wrong id would attach one patient's prescription to another's scan.
    """
    if len(results) != len(document_ids):
        raise ValueError("results and document_ids must be the same length")
    texts = source_texts or [None] * len(results)
    if len(texts) != len(results):
        raise ValueError("source_texts must match results in length")

    entries: list[TimelineEntry] = []
    unreadable: list[str] = []
    warnings: list[str] = []

    for result, document_id, text in zip(results, document_ids, texts):
        if result.mean_confidence < UNREADABLE_CONFIDENCE or not result.entities:
            unreadable.append(document_id)
            continue
        entry, entry_warnings = _to_entry(result, document_id, today=today, source_text=text)
        entries.append(entry)
        warnings.extend(entry_warnings)

    deduped, notes = deduplicate(entries)
    for note in notes:
        warnings.append(f"merged {note.absorbed} into {note.kept}: {note.reason}")

    ordered = order(deduped)

    # Interactions are checked across the WHOLE medication list, not per document —
    # the dangerous pair is usually two drugs from two different prescribers.
    # `check()` wants a bare drug name (brand or generic), not a full prescription
    # line, so we pull the name out the same way de-duplication does.
    drugs = [
        name
        for entry in ordered
        for e in entry.entities if e.kind == "medication"
        for name in [primary_drug_name(e.text)]
        if name
    ]
    interactions = check(drugs) if drugs else []

    readable = [r for r in results if r.mean_confidence >= UNREADABLE_CONFIDENCE]
    mean_confidence = (
        sum(r.mean_confidence for r in readable) / len(readable) if readable else 0.0
    )

    # Abnormal-value highlighting reads the coded lab entities already sitting
    # in the ordered timeline — no re-extraction, no re-OCR.
    interim_bundle = DocumentBundle(
        session_id=session_id, timeline=ordered, documents_processed=len(results),
    )
    abnormal_values, abnormal_warnings = flag_abnormal_values(interim_bundle)
    warnings.extend(abnormal_warnings)

    if unreadable:
        warnings.append(f"{len(unreadable)} document(s) unreadable — offer the patient a retake")

    return DocumentBundle(
        session_id=session_id,
        timeline=ordered,
        abnormal_values=abnormal_values,
        interactions=interactions,
        documents_processed=len(results),
        unreadable=unreadable,
        mean_confidence=mean_confidence,
        warnings=warnings,
    )
