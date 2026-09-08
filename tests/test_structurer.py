"""Tests for extraction/structurer.py.

These are the *extra* extraction tests beyond ml/tests/test_pipeline.py — the
team's regression gate, which this file must stay consistent with. Everything
here uses hand-written `OcrResult` fixtures over realistic whole documents, no
Tesseract dependency, per the brief.
"""

from __future__ import annotations

from sanjeevani_ml.extraction.structurer import extract_dates, structure
from sanjeevani_ml.schemas import ExtractedEntity, ExtractionResult, OcrResult

PHC_OPD_SLIP = """
PRIMARY HEALTH CENTRE, KOPARKHAIRANE
Navi Mumbai - 400709
OPD Slip                       Reg No: 2026/OPD/44817
Patient: Ramesh K.   Age/Sex: 54/M      Date: 03/04/2026

Vitals: BP 140/90 mmHg   Pulse 88 bpm   Temp 98.6 F
SpO2 97%   Weight 78 kg   Height 168 cm

Dx: Type 2 Diabetes Mellitus, Hypertension

Rx
1. Tab. Glycomet 500mg      1-0-1   x 30 days   (after food)
2. Tab. Telma 40mg          1-0-0   OD  x 30 days
3. Cap. Becosules           0-0-1   HS  x 10 days

Review after 1 month.
"""

LAB_REPORT = """
SUNRISE DIAGNOSTICS CENTRE
Lab No: 88231          Collected: 13/03/2026        Reported: 14/03/2026

TEST                        RESULT      UNIT            REFERENCE RANGE
Haemoglobin                 11.2        g/dL            13.0 - 17.0
HDL Cholesterol              38         mg/dL           40 - 60
LDL Cholesterol             142         mg/dL           0 - 100
"""


def _ocr(text: str, mean_confidence: float = 0.9, engine: str = "tesseract-5.3.0") -> OcrResult:
    return OcrResult(text=text, mean_confidence=mean_confidence, engine=engine)


def _by_kind(result: ExtractionResult, kind: str) -> list[ExtractedEntity]:
    return [e for e in result.entities if e.kind == kind]


# ---------------------------------------------------------------------------
# Shape / typing guarantees
# ---------------------------------------------------------------------------


def test_structure_returns_an_extraction_result() -> None:
    result = structure(_ocr(PHC_OPD_SLIP))
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) > 0
    assert all(isinstance(e, ExtractedEntity) for e in result.entities)


def test_no_entity_ever_carries_a_code() -> None:
    # Coding is terminology/mapper.py's job. structure() must never guess one.
    entities = structure(_ocr(PHC_OPD_SLIP)).entities + structure(_ocr(LAB_REPORT)).entities
    assert all(e.code is None for e in entities)


def test_fhir_is_empty_until_the_build_step() -> None:
    assert structure(_ocr(PHC_OPD_SLIP)).fhir == {}


def test_engine_is_carried_through_from_ocr() -> None:
    assert structure(_ocr(PHC_OPD_SLIP, engine="tesseract-5.3.0")).engine == "tesseract-5.3.0"


def test_every_entity_confidence_never_exceeds_ocr_confidence() -> None:
    ocr = _ocr(PHC_OPD_SLIP, mean_confidence=0.87)
    for entity in structure(ocr).entities:
        assert 0.0 <= entity.confidence <= 0.87


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_prescription_beats_diagnosis_in_classification() -> None:
    # PHC_OPD_SLIP has both a Dx line and drug lines — medications win.
    assert structure(_ocr(PHC_OPD_SLIP)).suggested_type == "prescription"


def test_lab_report_classifies_as_lab() -> None:
    assert structure(_ocr(LAB_REPORT)).suggested_type == "lab"


def test_bare_diagnosis_with_no_drugs_or_labs_classifies_as_condition() -> None:
    result = structure(_ocr("Dx: Seasonal Allergic Rhinitis"))
    assert result.suggested_type == "condition"


# ---------------------------------------------------------------------------
# extract_dates()
# ---------------------------------------------------------------------------


def test_extract_dates_prefers_the_labelled_date_over_an_unlabelled_one() -> None:
    text = "Ref No 03/04/2026 12345\nDate: 12/06/2026"
    iso, _ = extract_dates(text)
    assert iso == "2026-06-12"


def test_extract_dates_returns_none_and_zero_for_no_date() -> None:
    assert extract_dates("No dates in this line at all.") == (None, 0.0)


def test_clinical_date_field_matches_extract_dates() -> None:
    result = structure(_ocr(LAB_REPORT))
    iso, _ = extract_dates(LAB_REPORT)
    # Both dates in this fixture are labelled ("Collected:"/"Reported:"); ties
    # go to the first match in reading order, which is "Collected".
    assert result.clinical_date == iso == "2026-03-13"


# ---------------------------------------------------------------------------
# Facility
# ---------------------------------------------------------------------------


def test_facility_field_and_entity_agree() -> None:
    result = structure(_ocr(PHC_OPD_SLIP))
    facility_entities = _by_kind(result, "facility")
    assert len(facility_entities) == 1
    assert result.facility == facility_entities[0].text
    assert "PRIMARY HEALTH CENTRE" in result.facility.upper()


# ---------------------------------------------------------------------------
# Diagnoses
# ---------------------------------------------------------------------------


def test_diagnosis_line_splits_into_one_entity_per_condition() -> None:
    diagnoses = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "diagnosis")
    assert [d.text for d in diagnoses] == ["Type 2 Diabetes Mellitus", "Hypertension"]


# ---------------------------------------------------------------------------
# Medications — plain-language dosing
# ---------------------------------------------------------------------------


def test_medication_text_starts_with_the_line_as_written() -> None:
    meds = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "medication")
    assert meds[0].text.startswith("1. Tab. Glycomet 500mg")


def test_medication_dosing_reads_in_plain_language() -> None:
    meds = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "medication")
    glycomet = next(m for m in meds if "Glycomet" in m.text)
    assert "1 tablet in the morning" in glycomet.text
    assert "1 tablet at night" in glycomet.text
    assert "for 30 days" in glycomet.text


def test_od_only_schedule_reads_naturally() -> None:
    meds = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "medication")
    telma = next(m for m in meds if "Telma" in m.text)
    assert "1 tablet in the morning" in telma.text


def test_half_tablet_dose_reads_as_half_a_tablet() -> None:
    ocr = _ocr("Tab. Amlong 5mg 1/2-0-1/2 x 30 days")
    meds = _by_kind(structure(ocr), "medication")
    assert "half a tablet in the morning" in meds[0].text
    assert "half a tablet at night" in meds[0].text


def test_medication_strength_is_captured_as_value_and_unit() -> None:
    meds = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "medication")
    glycomet = next(m for m in meds if "Glycomet" in m.text)
    assert glycomet.value == 500.0
    assert glycomet.unit == "mg"


# ---------------------------------------------------------------------------
# Lab rows
# ---------------------------------------------------------------------------


def test_lab_entity_text_is_the_bare_analyte_name() -> None:
    labs = _by_kind(structure(_ocr(LAB_REPORT)), "lab")
    by_analyte = {lab.text: lab for lab in labs}
    assert "Haemoglobin" in by_analyte
    assert by_analyte["Haemoglobin"].value == 11.2
    assert by_analyte["Haemoglobin"].unit == "g/dL"
    assert by_analyte["Haemoglobin"].reference_range == "13.0-17.0"


def test_hdl_and_ldl_remain_distinct_lab_entities() -> None:
    labs = _by_kind(structure(_ocr(LAB_REPORT)), "lab")
    by_analyte = {lab.text: lab.value for lab in labs}
    assert by_analyte["HDL Cholesterol"] == 38.0
    assert by_analyte["LDL Cholesterol"] == 142.0


def test_lab_row_with_parenthesized_reference_range() -> None:
    ocr = _ocr("Fasting Blood Sugar    142   mg/dL   (70-100)")
    labs = _by_kind(structure(ocr), "lab")
    assert labs[0].reference_range == "70-100"
    assert labs[0].value == 142.0


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------


def test_bp_entity_holds_the_raw_pair_as_a_string() -> None:
    vitals = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "vital")
    bp = next(v for v in vitals if v.unit == "mmHg")
    assert bp.value == "140/90"
    assert "systolic=140" in bp.text and "diastolic=90" in bp.text


def test_all_six_single_value_vitals_present() -> None:
    vitals = _by_kind(structure(_ocr(PHC_OPD_SLIP)), "vital")
    by_unit = {v.unit: v.value for v in vitals}
    assert by_unit["bpm"] == 88.0
    assert by_unit["%"] == 97.0
    assert by_unit["F"] == 98.6
    assert by_unit["kg"] == 78.0
    assert by_unit["cm"] == 168.0


def test_vital_confidence_reflects_ocr_confidence_directly() -> None:
    high = structure(_ocr(PHC_OPD_SLIP, mean_confidence=0.95))
    low = structure(_ocr(PHC_OPD_SLIP, mean_confidence=0.4))
    high_bp = next(e for e in high.entities if e.kind == "vital" and e.unit == "mmHg")
    low_bp = next(e for e in low.entities if e.kind == "vital" and e.unit == "mmHg")
    assert low_bp.confidence < high_bp.confidence
    assert low_bp.confidence < 0.5


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def test_empty_document_yields_no_entities_and_a_warning() -> None:
    result = structure(_ocr(""))
    assert result.entities == []
    assert result.mean_confidence == 0.0
    assert result.warnings


def test_document_with_no_recognisable_patterns_warns_but_does_not_crash() -> None:
    result = structure(_ocr("This page intentionally left blank."))
    assert result.entities == []
    assert result.warnings


def test_zero_ocr_confidence_zeroes_every_entity_confidence() -> None:
    result = structure(_ocr(PHC_OPD_SLIP, mean_confidence=0.0))
    assert len(result.entities) > 0  # patterns still match — the document still has shape
    assert all(e.confidence == 0.0 for e in result.entities)  # but nothing is trustworthy