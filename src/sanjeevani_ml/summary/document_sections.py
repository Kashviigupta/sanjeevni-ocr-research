"""The parts of the physician summary that can be built from paperwork alone.

Module C's full summary fuses Mahek's InterviewTranscript with this module's output.
Two of its ten sections, though, need no interview data at all — everything they say
comes from documents the patient photographed, which this module already has fully
processed by the time it runs (timeline built, terminology coded, abnormal values
flagged). Building these now means Module C only has to plug in a transcript later,
not build these sections from scratch once one exists.

Both sections here are drafts, like everything else in this pipeline: every line
names the document it came from, nothing is phrased as a diagnosis, and a document
this module cannot summarise (no lab/medication entities to speak of) produces an
honest "insufficient documentation" line rather than an empty, unexplained gap.

Bilingual output is a separate, not-yet-started piece of work (see project notes).
`SummarySection.content` here is English only (`{"en": ...}`) until that lands —
this module does not fabricate a Hindi translation to fill the shape.
"""
from __future__ import annotations

from sanjeevani_ml.schemas import AbnormalValue, DocumentBundle, SummarySection, TimelineEntry
from sanjeevani_ml.summary.translations import translate_template

#: Below this, a medication's coding confidence is too low to state its identity as
#: settled fact in a physician-facing sentence — it is still shown, but hedged with
#: "as read" rather than stated plainly. Same status as every other threshold in this
#: pipeline: named, justified, meant to be tuned against real documents rather than
#: reasoned about in the abstract.
_LOW_CONFIDENCE = 0.6


def _formatted_date(entry: TimelineEntry) -> str:
    return entry.date if entry.date else "undated"


def _medication_lines(bundle: DocumentBundle, *, lang: str = "en") -> tuple[list[str], list[str]]:
    """One line per medication entity found across the whole timeline, newest first.

    Returns `(lines, provenance_document_ids)`. A drug is listed once per document
    that mentions it — we do not silently collapse "same drug, different
    prescriptions" here, because the timeline's own de-duplication has already
    removed genuine duplicate scans; anything still distinct at this point is a
    separate prescribing event and the physician should see each one.

    The drug name and date are DATA, identical in every language. Only the
    confidence-hedge phrase is a template, so it is the only part that changes
    between `lang="en"` and `lang="hi"`.
    """
    lines: list[str] = []
    provenance: list[str] = []
    for entry in reversed(bundle.timeline):  # newest first — most relevant to a physician
        for entity in entry.entities:
            if entity.kind != "medication":
                continue
            if entity.confidence >= _LOW_CONFIDENCE:
                hedge = ""
            else:
                hedge_template = " (as read; low confidence)"
                hedge = hedge_template if lang == "en" else f" {translate_template('(as read; low confidence)')}"
            lines.append(f"{entity.text}{hedge} — {_formatted_date(entry)}")
            if entry.document_id:
                provenance.append(entry.document_id)
    return lines, provenance


def _allergy_document_lines(bundle: DocumentBundle) -> tuple[list[str], list[str]]:
    """Any document extraction itself classified as an allergy record.

    There is no dedicated `allergy` entity kind in the current schema (only a
    document-level `suggested_type`) — so this is deliberately coarse: it names
    the document, not a structured allergen list, because we have nothing more
    specific to say without over-claiming.
    """
    lines: list[str] = []
    provenance: list[str] = []
    for entry in bundle.timeline:
        if entry.kind == "allergy":
            lines.append(f"Allergy record on file: \"{entry.title}\" — {_formatted_date(entry)}")
            if entry.document_id:
                provenance.append(entry.document_id)
    return lines, provenance


def build_drug_allergy_section(bundle: DocumentBundle) -> SummarySection:
    """The document-derived half of "Drug & allergy history".

    This is explicitly NOT the whole section — Module C must still ask the patient
    directly (via the interview) whether they take anything not represented in a
    scanned document, and must surface it as a contradiction if the two disagree.
    This function only ever states what the paperwork itself shows.

    Bilingual: drug names, dates, and confidence hedges are DATA — identical in
    both languages. Only the surrounding template sentences translate, via the
    curated (not machine-translated) table in `translations.py`.
    """
    med_lines_en, med_provenance = _medication_lines(bundle, lang="en")
    med_lines_hi, _ = _medication_lines(bundle, lang="hi")
    allergy_lines, allergy_provenance = _allergy_document_lines(bundle)

    lines_en = [*med_lines_en, *allergy_lines]
    lines_hi = [*med_lines_hi, *allergy_lines]
    provenance = list(dict.fromkeys([*med_provenance, *allergy_provenance]))  # dedup, keep order

    if not lines_en:
        empty_template = "No medications or allergy records found among the documents provided."
        content = {"en": empty_template, "hi": translate_template(empty_template)}
        confidence = 0.0
    else:
        content = {"en": "\n".join(lines_en), "hi": "\n".join(lines_hi)}
        confidences = [
            e.confidence for entry in bundle.timeline for e in entry.entities
            if e.kind == "medication"
        ]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return SummarySection(
        heading="drugs_allergy",
        content=content,
        provenance=provenance,
        confidence=confidence,
    )


def build_prior_investigations_section(bundle: DocumentBundle) -> SummarySection:
    """"Prior investigations summary" — labs, imaging, and flagged abnormal values.

    Leads with abnormal values, per the platform's "concise or it does not get
    read" rule: a physician with two minutes should see what's actually wrong
    before wading through every normal result.

    Bilingual: section headers ("Abnormal results:", etc.) are curated templates
    that translate; analyte names, values, units, dates, and drug names are data
    and appear identically in both languages.
    """
    lines: list[str] = []
    provenance: list[str] = []

    if bundle.abnormal_values:
        lines.append(("template", "Abnormal results:"))
        for abnormal in bundle.abnormal_values:
            lines.append(("data",
                f"  - {abnormal.analyte}: {abnormal.value} {abnormal.unit or ''} "
                f"({abnormal.direction}, reference {abnormal.reference_range})".strip()
            ))
            if abnormal.source_document_id:
                provenance.append(abnormal.source_document_id)

    lab_entries = [e for e in bundle.timeline if e.kind == "lab"]
    if lab_entries:
        if lines:
            lines.append(("data", ""))
        lines.append(("template", "All investigations on file:"))
        for entry in lab_entries:
            lines.append(("data", f"  - {entry.title} — {_formatted_date(entry)}"))
            if entry.document_id:
                provenance.append(entry.document_id)

    if bundle.interactions:
        if lines:
            lines.append(("data", ""))
        lines.append(("template", "Drug interaction warnings:"))
        for alert in bundle.interactions:
            lines.append(("data", f"  - {alert.drug_a} + {alert.drug_b}: {alert.description} ({alert.severity})"))

    if not lines:
        empty_template = "No prior investigation reports found among the documents provided."
        content = {"en": empty_template, "hi": translate_template(empty_template)}
        confidence = 0.0
    else:
        def render(lang: str) -> str:
            return "\n".join(
                (translate_template(text) if lang == "hi" else text) if kind == "template" else text
                for kind, text in lines
            )
        content = {"en": render("en"), "hi": render("hi")}
        confidence = bundle.mean_confidence

    return SummarySection(
        heading="prior_investigations",
        content=content,
        provenance=list(dict.fromkeys(provenance)),
        confidence=confidence,
    )


def build_document_only_sections(bundle: DocumentBundle) -> list[SummarySection]:
    """Both document-derived sections, ready for Module C to slot into a full summary."""
    return [
        build_drug_allergy_section(bundle),
        build_prior_investigations_section(bundle),
    ]
