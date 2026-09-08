"""Abnormal-value highlighting tests.

Every branch of the decision rule gets its own fixture: normal, high, low, boundary,
missing document range (curated fallback), unit mismatch, garbled printed range, and
no range anywhere. Nothing here is a synthetic string invented to match the code —
each is what an actual lab printout row looks like once OCR/extraction has run.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from sanjeevani_ml.schemas import ExtractedEntity, TerminologyCode
from sanjeevani_ml.timeline.abnormal import evaluate_lab_entity, flag_abnormal_values
from sanjeevani_ml.schemas import DocumentBundle, TimelineEntry

#: Stand-in curated table row, patched in per test so these tests do not depend on
#: what ml/data/lab_terms.csv actually contains yet (that column may not exist).
FBS_CURATED_ROW = {
    "code": "1558-6", "term": "Fasting Blood Sugar", "display": "Fasting glucose",
    "reference_range": "70-100", "unit": "mg/dL",
}


def _lab(text, value, unit=None, ref_range=None, code=None) -> ExtractedEntity:
    return ExtractedEntity(
        kind="lab", text=text, value=value, unit=unit, reference_range=ref_range,
        code=code, confidence=0.9,
    )


def _with_curated(rows):
    return patch("sanjeevani_ml.timeline.abnormal.load_lab_terms", return_value=rows)


class TestDocumentRangeWins:
    def test_a_value_inside_the_documents_own_range_is_normal(self):
        entity = _lab("Fasting Blood Sugar", 85, "mg/dL", "70-100")
        with _with_curated([FBS_CURATED_ROW]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is None
        assert not warnings

    def test_a_value_above_the_documents_range_is_flagged_high(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL", "70-100")
        with _with_curated([FBS_CURATED_ROW]):
            result, _ = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None
        assert result.direction == "high"
        assert result.value == 142

    def test_a_value_below_the_documents_range_is_flagged_low(self):
        entity = _lab("Haemoglobin", 8.0, "g/dL", "12-15")
        with _with_curated([]):
            result, _ = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None
        assert result.direction == "low"

    def test_boundary_values_are_normal_not_abnormal(self):
        """Inclusive range: 100 itself is still within '70-100'."""
        entity = _lab("Fasting Blood Sugar", 100, "mg/dL", "70-100")
        with _with_curated([FBS_CURATED_ROW]):
            result, _ = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is None

    def test_the_source_document_is_recorded_on_the_flag(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL", "70-100")
        with _with_curated([FBS_CURATED_ROW]):
            result, _ = evaluate_lab_entity(entity, document_id="lab-mar-14")
        assert result.source_document_id == "lab-mar-14"


class TestCuratedFallback:
    def test_missing_document_range_falls_back_to_curated(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL", ref_range=None)
        with _with_curated([FBS_CURATED_ROW]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None
        assert result.direction == "high"
        assert not warnings  # a clean fallback, nothing to warn about

    def test_no_range_anywhere_reports_no_finding_and_warns(self):
        """An absent AbnormalValue must never be misread as 'checked and normal'."""
        entity = _lab("Some Rare Analyte", 999, "units", ref_range=None)
        with _with_curated([]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is None
        assert any("no reference range" in w for w in warnings)


class TestUnitGuard:
    def test_a_unit_mismatch_blocks_comparison_rather_than_converting(self):
        entity = _lab("Fasting Blood Sugar", 5.5, "mmol/L", "70-100")
        with _with_curated([FBS_CURATED_ROW]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is None
        assert any("unit" in w.lower() for w in warnings)

    def test_equivalent_unit_spellings_are_not_treated_as_a_mismatch(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dl", "70-100")
        row = {**FBS_CURATED_ROW, "unit": "mg/dL"}
        with _with_curated([row]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None
        assert not any("unit" in w.lower() for w in warnings)


class TestGarbledRangeGuard:
    def test_an_implausibly_narrow_printed_range_falls_back_to_curated(self):
        """'10-100' next to a curated '70-100' looks like a dropped '7'."""
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL", "10-100")
        with _with_curated([FBS_CURATED_ROW]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None
        assert result.reference_range == "70.0-100.0"
        assert any("implausible" in w for w in warnings)

    def test_a_plausible_disagreement_is_kept_and_warned_not_overridden(self):
        """A different but sane range (assay variation) is real data, not garbage."""
        entity = _lab("Fasting Blood Sugar", 95, "mg/dL", "60-95")
        with _with_curated([FBS_CURATED_ROW]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is None  # 95 is within the document's own 60-95
        assert any("differs from the curated range" in w for w in warnings)

    def test_an_inverted_range_is_never_used(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL", "100-70")
        with _with_curated([FBS_CURATED_ROW]):
            result, _ = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None  # falls back to curated, since 100-70 doesn't parse
        assert result.reference_range == "70.0-100.0"


class TestNonNumericAndDegenerate:
    def test_a_non_numeric_lab_value_is_never_force_parsed(self):
        entity = _lab("HIV", "Non-reactive", ref_range=None)
        with _with_curated([]):
            result, warnings = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is None
        assert warnings == []

    def test_an_unparseable_range_string_is_treated_as_absent(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL", "abnormal")
        with _with_curated([FBS_CURATED_ROW]):
            result, _ = evaluate_lab_entity(entity, document_id="lab-1")
        assert result is not None
        assert result.reference_range == "70.0-100.0"


class TestBundleLevel:
    def test_flags_across_a_whole_bundle_not_just_one_entity(self):
        entry = TimelineEntry(
            date="2026-03-14", kind="lab", title="Lab", document_id="lab-1",
            entities=[
                _lab("Fasting Blood Sugar", 142, "mg/dL", "70-100"),
                _lab("Haemoglobin", 11.2, "g/dL", "12-15"),
                _lab("Total Cholesterol", 180, "mg/dL", "0-200"),
            ],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        with _with_curated([]):
            abnormal, _ = flag_abnormal_values(bundle)
        directions = {a.analyte: a.direction for a in abnormal}
        assert directions == {"Fasting Blood Sugar": "high", "Haemoglobin": "low"}

    def test_non_lab_entities_are_ignored(self):
        entry = TimelineEntry(
            date="2026-04-02", kind="prescription", title="Rx", document_id="rx-1",
            entities=[ExtractedEntity(kind="medication", text="Tab. Glycomet 500mg", confidence=0.9)],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        with _with_curated([]):
            abnormal, warnings = flag_abnormal_values(bundle)
        assert abnormal == []
        assert warnings == []

    def test_an_empty_bundle_returns_no_findings_and_no_crash(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        abnormal, warnings = flag_abnormal_values(bundle)
        assert abnormal == []
        assert warnings == []
