"""Tests for the document-only Module C pieces: summary sections built purely from
DocumentBundle, and the documentation-gaps report. Neither needs an InterviewTranscript.
"""
from __future__ import annotations

from sanjeevani_ml.schemas import (
    AbnormalValue,
    DocumentBundle,
    ExtractedEntity,
    InteractionAlert,
    TimelineEntry,
)
from sanjeevani_ml.summary.document_sections import (
    build_document_only_sections,
    build_drug_allergy_section,
    build_prior_investigations_section,
)
from sanjeevani_ml.summary.gaps import build_documentation_gaps_report


def _med(text, confidence=0.9) -> ExtractedEntity:
    return ExtractedEntity(kind="medication", text=text, confidence=confidence)


def _lab(text, confidence=0.9) -> ExtractedEntity:
    return ExtractedEntity(kind="lab", text=text, confidence=confidence)


class TestDrugAllergySection:
    def test_lists_medications_from_the_timeline(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[_med("Tab. Glycomet 500mg"), _med("Tab. Amlong 5mg")],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert section.heading == "drugs_allergy"
        assert "Glycomet" in section.content["en"]
        assert "Amlong" in section.content["en"]

    def test_names_the_source_document_in_provenance(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[_med("Tab. Glycomet 500mg")],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert "rx-1" in section.provenance

    def test_low_confidence_medication_is_hedged_in_text(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[_med("Tab. Illegible 500mg", confidence=0.3)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert "low confidence" in section.content["en"]

    def test_an_allergy_document_is_named_not_fabricated_into_a_list(self):
        entry = TimelineEntry(
            date="2026-01-01", kind="allergy", title="Penicillin allergy card",
            document_id="allergy-1",
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert "Penicillin allergy card" in section.content["en"]
        assert "allergy-1" in section.provenance

    def test_no_documents_produces_an_honest_empty_statement_not_a_blank(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        section = build_drug_allergy_section(bundle)
        assert "No medications" in section.content["en"]
        assert section.confidence == 0.0

    def test_never_phrases_anything_as_a_diagnosis(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[_med("Tab. Glycomet 500mg")],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        text_lower = section.content["en"].lower()
        assert "diagnosis" not in text_lower
        assert "likely" not in text_lower


class TestPriorInvestigationsSection:
    def test_leads_with_abnormal_results(self):
        bundle = DocumentBundle(
            session_id="s1",
            timeline=[TimelineEntry(date="2026-03-14", kind="lab", title="Lab", document_id="lab-1")],
            abnormal_values=[
                AbnormalValue(
                    analyte="Fasting Blood Sugar", value=142, unit="mg/dL",
                    reference_range="70.0-100.0", direction="high", source_document_id="lab-1",
                )
            ],
        )
        section = build_prior_investigations_section(bundle)
        assert section.content["en"].startswith("Abnormal results:")
        assert "Fasting Blood Sugar" in section.content["en"]

    def test_lists_all_lab_documents_even_without_abnormal_values(self):
        entry = TimelineEntry(date="2026-03-14", kind="lab", title="CBC Report", document_id="lab-1")
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_prior_investigations_section(bundle)
        assert "CBC Report" in section.content["en"]

    def test_includes_interaction_warnings(self):
        bundle = DocumentBundle(
            session_id="s1", timeline=[],
            interactions=[
                InteractionAlert(severity="major", drug_a="warfarin", drug_b="aspirin", description="Bleeding risk")
            ],
        )
        section = build_prior_investigations_section(bundle)
        assert "warfarin" in section.content["en"]
        assert "Bleeding risk" in section.content["en"]

    def test_no_investigations_produces_an_honest_empty_statement(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        section = build_prior_investigations_section(bundle)
        assert "No prior investigation" in section.content["en"]

    def test_provenance_includes_every_source_document(self):
        bundle = DocumentBundle(
            session_id="s1",
            timeline=[TimelineEntry(date="2026-03-14", kind="lab", title="Lab", document_id="lab-1")],
            abnormal_values=[
                AbnormalValue(
                    analyte="HbA1c", value=7.8, unit="%", reference_range="4.0-5.6",
                    direction="high", source_document_id="lab-1",
                )
            ],
        )
        section = build_prior_investigations_section(bundle)
        assert section.provenance.count("lab-1") == 1  # deduped, not repeated


class TestBuildDocumentOnlySections:
    def test_returns_both_sections_with_correct_headings(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        sections = build_document_only_sections(bundle)
        headings = {s.heading for s in sections}
        assert headings == {"drugs_allergy", "prior_investigations"}


class TestDocumentationGapsReport:
    def test_reports_unreadable_documents(self):
        bundle = DocumentBundle(
            session_id="s1", timeline=[], documents_processed=2, unreadable=["blurry-1"],
        )
        report = build_documentation_gaps_report(bundle)
        assert "blurry-1" in report.documents_unreadable
        assert "could not be read" in report.as_plain_text()

    def test_flags_low_confidence_entities_by_name(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="lab", title="Lab", document_id="lab-1",
            entities=[_lab("Suspicious Analyte", confidence=0.3)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry], documents_processed=1)
        report = build_documentation_gaps_report(bundle)
        assert "Suspicious Analyte" in report.uncertain_entities

    def test_a_confident_entity_is_never_listed_as_uncertain(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="lab", title="Lab", document_id="lab-1",
            entities=[_lab("Confident Analyte", confidence=0.95)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry], documents_processed=1)
        report = build_documentation_gaps_report(bundle)
        assert "Confident Analyte" not in report.uncertain_entities

    def test_entirely_undocumented_when_every_document_is_unreadable(self):
        bundle = DocumentBundle(
            session_id="s1", timeline=[], documents_processed=1, unreadable=["blurry-1"],
        )
        report = build_documentation_gaps_report(bundle)
        assert report.entirely_undocumented
        assert "No usable clinical information" in report.as_plain_text()

    def test_a_clean_bundle_is_not_flagged_as_entirely_undocumented(self):
        entry = TimelineEntry(date="2026-04-02", kind="lab", title="Lab", entities=[_lab("X", 0.9)])
        bundle = DocumentBundle(session_id="s1", timeline=[entry], documents_processed=1)
        report = build_documentation_gaps_report(bundle)
        assert not report.entirely_undocumented

    def test_carries_forward_warnings_from_earlier_pipeline_stages(self):
        bundle = DocumentBundle(
            session_id="s1", timeline=[], documents_processed=1,
            warnings=["lab-1: no reference range on the document or in the curated table"],
        )
        report = build_documentation_gaps_report(bundle)
        assert len(report.unresolved_warnings) == 1
        assert "manual review" in report.as_plain_text()

    def test_plain_text_is_short_enough_for_a_two_minute_consultation(self):
        """The platform's own rule: concise or it does not get read."""
        entry = TimelineEntry(
            date="2026-04-02", kind="lab", title="Lab", document_id="lab-1",
            entities=[_lab("A", 0.3), _lab("B", 0.3), _lab("C", 0.3), _lab("D", 0.3)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry], documents_processed=1)
        report = build_documentation_gaps_report(bundle)
        assert len(report.as_plain_text()) < 400
