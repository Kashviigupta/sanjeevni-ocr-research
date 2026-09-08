"""Tests for the curated English-Hindi template translation table and its wiring
into the document-only summary sections.

These do NOT assert the Hindi is linguistically correct — that verification is
a human, native-speaker task per the platform's AYUSH-content discipline, not
something a test can check. What these tests guarantee is the MECHANISM: every
template has a translation, drug/date/number data never gets translated, and an
untranslated template degrades to English rather than crashing or guessing.
"""
from __future__ import annotations

from sanjeevani_ml.schemas import AbnormalValue, DocumentBundle, ExtractedEntity, TimelineEntry
from sanjeevani_ml.summary.document_sections import (
    build_drug_allergy_section,
    build_prior_investigations_section,
)
from sanjeevani_ml.summary.translations import TEMPLATES, translate_template


class TestTranslateTemplate:
    def test_a_curated_template_returns_its_hindi_entry(self):
        assert translate_template("Abnormal results:") == TEMPLATES["Abnormal results:"]

    def test_an_uncurated_string_falls_back_to_english_not_a_guess(self):
        """No confident match is a correct, honest outcome — same as never-invent-a-code."""
        result = translate_template("Some sentence not in the table")
        assert result == "Some sentence not in the table"

    def test_every_template_entry_is_actually_different_from_its_key(self):
        """Catches an accidental copy-paste that would silently 'translate' to English."""
        for english, hindi in TEMPLATES.items():
            assert english != hindi


class TestBilingualDrugAllergySection:
    def test_both_language_keys_are_present(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[ExtractedEntity(kind="medication", text="Tab. Glycomet 500mg", confidence=0.9)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert "en" in section.content and "hi" in section.content

    def test_drug_names_are_identical_in_both_languages(self):
        """A physician reading Hindi still expects 'Glycomet 500mg', not a translation."""
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[ExtractedEntity(kind="medication", text="Tab. Glycomet 500mg", confidence=0.9)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert "Glycomet" in section.content["en"]
        assert "Glycomet" in section.content["hi"]

    def test_the_empty_case_is_translated(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        section = build_drug_allergy_section(bundle)
        assert section.content["hi"] != section.content["en"]
        assert section.content["hi"] == TEMPLATES["No medications or allergy records found among the documents provided."]

    def test_low_confidence_hedge_is_translated_in_hindi_but_not_english(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[ExtractedEntity(kind="medication", text="Tab. Illegible 500mg", confidence=0.3)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        section = build_drug_allergy_section(bundle)
        assert "low confidence" in section.content["en"]
        assert "low confidence" not in section.content["hi"]


class TestBilingualPriorInvestigationsSection:
    def test_both_language_keys_are_present(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        section = build_prior_investigations_section(bundle)
        assert "en" in section.content and "hi" in section.content

    def test_the_abnormal_results_header_is_translated(self):
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
        assert section.content["hi"].startswith(TEMPLATES["Abnormal results:"])

    def test_the_analyte_name_and_numbers_are_unchanged_in_hindi(self):
        """The value, unit, and analyte name are data — never translated."""
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
        assert "Fasting Blood Sugar" in section.content["hi"]
        assert "142" in section.content["hi"]
        assert "mg/dL" in section.content["hi"]
