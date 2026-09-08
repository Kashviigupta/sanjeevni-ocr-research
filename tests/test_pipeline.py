"""Smoke tests for the extraction pipeline.

These run without Tesseract installed — they feed text straight in, so Kashvi is never
blocked on Mahek's local setup and CI does not need the binary.

**When you add a capability, add a test with a REAL document fixture**, not a synthetic
string that happens to match the regex you just wrote. A regex tested only against the
example it was written for tells you nothing.
"""
from __future__ import annotations

from sanjeevani_ml.extraction.structurer import extract_dates, structure
from sanjeevani_ml.fhir.builders import build
from sanjeevani_ml.interactions.checker import check, summarize
from sanjeevani_ml.schemas import OcrResult
from sanjeevani_ml.terminology.mapper import code_result, generic_name

LAB_REPORT = """
City Diagnostics Laboratory
123 MG Road, Pune
Patient: [redacted]        Collected on: 14/03/2026

Fasting Blood Sugar    142   mg/dL   (70-100)
HbA1c                   7.8  %       (4.0-5.6)
Total Cholesterol      210   mg/dL   (0-200)
Haemoglobin             11.2 g/dL    (12-15)
"""

PRESCRIPTION = """
Dr. A. Sharma, MBBS MD
Sunrise Clinic, Nashik
Date: 02/04/2026

Rx
Tab. Glycomet 500mg 1-0-1 x 30 days
Tab. Amlong 5mg 1-0-0 x 30 days
Cap. Omez 20mg 1-0-0 x 14 days
"""


def _ocr(text: str) -> OcrResult:
    return OcrResult(text=text, mean_confidence=0.92, engine="test-fixture")


class TestLabExtraction:
    def test_classifies_a_lab_report(self):
        assert structure(_ocr(LAB_REPORT)).suggested_type == "lab"

    def test_pulls_analyte_value_and_unit(self):
        labs = {e.text.lower(): e for e in structure(_ocr(LAB_REPORT)).entities if e.kind == "lab"}
        assert "fasting blood sugar" in labs
        assert labs["fasting blood sugar"].value == 142
        assert labs["fasting blood sugar"].unit == "mg/dL"

    def test_keeps_the_reference_range(self):
        labs = {e.text.lower(): e for e in structure(_ocr(LAB_REPORT)).entities if e.kind == "lab"}
        assert labs["fasting blood sugar"].reference_range == "70-100"

    def test_maps_to_loinc(self):
        result = code_result(structure(_ocr(LAB_REPORT)))
        fbs = next(e for e in result.entities if "fasting" in e.text.lower())
        assert fbs.code is not None
        assert fbs.code.system == "LOINC"
        assert fbs.code.code == "1558-6"

    def test_finds_the_facility(self):
        assert "City Diagnostics" in (structure(_ocr(LAB_REPORT)).facility or "")


class TestPrescriptionExtraction:
    def test_classifies_a_prescription(self):
        assert structure(_ocr(PRESCRIPTION)).suggested_type == "prescription"

    def test_finds_every_drug_line(self):
        meds = [e for e in structure(_ocr(PRESCRIPTION)).entities if e.kind == "medication"]
        assert len(meds) == 3

    def test_expands_indian_dosing_notation(self):
        """`1-0-1` must become something a patient can actually read."""
        meds = [e for e in structure(_ocr(PRESCRIPTION)).entities if e.kind == "medication"]
        glycomet = next(m for m in meds if "Glycomet" in m.text)
        assert "morning" in glycomet.text and "night" in glycomet.text

    def test_resolves_brand_to_generic(self):
        """Indian prescriptions are written in brands; the interaction table is generic."""
        assert generic_name("Glycomet") == "metformin"
        assert generic_name("Amlong") == "amlodipine"


class TestDates:
    def test_prefers_a_labelled_date(self):
        iso, conf = extract_dates(LAB_REPORT)
        assert iso == "2026-03-14"
        assert conf > 0.8

    def test_reads_dd_mm_not_mm_dd(self):
        """03/04/2026 is 3 April in India. Reading it as 4 March is a clinical error."""
        iso, _ = extract_dates("Date: 03/04/2026")
        assert iso == "2026-04-03"

    def test_rejects_an_impossible_date(self):
        assert extract_dates("Date: 45/13/2026")[0] is None


class TestFhir:
    def test_lab_report_becomes_a_diagnostic_report(self):
        result = code_result(structure(_ocr(LAB_REPORT)))
        build(result)
        assert result.fhir["resourceType"] == "DiagnosticReport"
        assert len(result.fhir["contained"]) >= 3

    def test_output_is_never_final(self):
        """Machine output is a draft. `final` would claim clinical truth we do not have."""
        result = code_result(structure(_ocr(LAB_REPORT)))
        build(result)
        assert result.fhir["status"] == "preliminary"

    def test_provenance_names_the_engine(self):
        result = code_result(structure(_ocr(LAB_REPORT)))
        build(result)
        urls = [e["url"] for e in result.fhir["extension"]]
        assert any("extraction-engine" in u for u in urls)
        assert any("extraction-confidence" in u for u in urls)

    def test_prescription_becomes_medication_requests(self):
        result = code_result(structure(_ocr(PRESCRIPTION)))
        build(result)
        assert result.fhir["resourceType"] == "Bundle"
        assert result.fhir["entry"][0]["resource"]["resourceType"] == "MedicationRequest"


class TestInteractions:
    def test_catches_a_major_interaction(self):
        alerts = check(["warfarin", "aspirin"])
        assert len(alerts) == 1
        assert alerts[0].severity == "major"

    def test_works_on_brand_names(self):
        """The whole point: a patient reads brands off their strip of tablets."""
        assert len(check(["Ecosprin", "Warf"])) == 1

    def test_orders_worst_first(self):
        alerts = check(["warfarin", "aspirin", "paracetamol"])
        assert alerts[0].severity == "major"

    def test_never_claims_safety(self):
        """"No interactions found" must not read as "this combination is safe"."""
        summary = summarize(check(["paracetamol"]))
        assert "safe" not in summary.lower()
        assert "not a complete" in summary.lower()


class TestDegradation:
    def test_empty_document_does_not_crash(self):
        """Degrade, never fail (spec principle 6) — the app falls back to manual entry."""
        result = structure(_ocr(""))
        assert result.mean_confidence == 0.0
        assert result.warnings

    def test_unparseable_text_still_returns_a_record(self):
        result = code_result(structure(_ocr("असंबंधित पाठ ####")))
        build(result)
        assert result.fhir["resourceType"] in ("DocumentReference", "DiagnosticReport")


class TestConfidence:
    def test_confidence_is_never_inflated(self):
        """Rule certainty multiplies OCR confidence — it can never exceed it."""
        ocr = _ocr(LAB_REPORT)
        result = structure(ocr)
        assert all(e.confidence <= ocr.mean_confidence + 1e-9 for e in result.entities)
