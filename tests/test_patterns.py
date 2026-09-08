"""Tests for extraction/patterns.py.

Every test runs against a *realistic whole document* fixture — a PHC OPD slip, a
private lab printout, a Hindi/English mixed slip — not against a string written
to match the regex under test. A regex tested only against the example it was
written for tells us nothing.

Fixtures are synthetic: invented patient names and values, no real PHI.
"""

from __future__ import annotations

from datetime import date

import pytest

from sanjeevani_ml.extraction.patterns import (
    BP_RE,
    DATE_DMY_RE,
    DOSE_SLOT_RE,
    DRUG_LINE_RE,
    DURATION_RE,
    FACILITY_RE,
    FREQUENCY_RE,
    HEIGHT_RE,
    LAB_ROW_RE,
    PULSE_RE,
    SPO2_RE,
    STRENGTH_RE,
    TEMPERATURE_RE,
    WEIGHT_RE,
    normalize_text,
    parse_date,
    parse_dose_notation,
)

# ---------------------------------------------------------------------------
# Fixtures: whole documents, as OCR would hand them over
# ---------------------------------------------------------------------------

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
4. Syp. Ascoril             5 ml    TDS x 5 days

Review after 1 month.
"""

LAB_REPORT = """
SUNRISE DIAGNOSTICS CENTRE
Lab No: 88231          Collected: 13/03/2026        Reported: 14/03/2026

TEST                        RESULT      UNIT            REFERENCE RANGE
Haemoglobin                 11.2        g/dL            13.0 - 17.0
Total Leucocyte Count       9800        cells/cumm      4000 - 11000
Fasting Blood Sugar         126         mg/dL           70 - 100
HbA1c                       7.8         %               4.0 - 5.6
Serum Creatinine            1.4         mg/dL           0.7 - 1.3
HDL Cholesterol             38          mg/dL           40 - 60
LDL Cholesterol             142         mg/dL           0 - 100
TSH                         5.9         mIU/L           0.4 - 4.0
"""

HINDI_MIXED_SLIP = """
सामुदायिक स्वास्थ्य केंद्र, वाशी
दिनांक: १५/०२/२०२६

बीपी: १३०/८५ mmHg    नाड़ी: ७६/min    वजन: ६८ किलो
तापमान: 99.2 F

Rx
गोली Metformin 500mg   १-०-१   x १० दिन
Tab. Pan 40            1-0-0   BD
"""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_folds_devanagari_digits_and_keeps_lines() -> None:
    out = normalize_text(HINDI_MIXED_SLIP)
    assert "15/02/2026" in out
    assert "130/85" in out
    # Line structure must survive: column-aligned layouts depend on it.
    assert len(out.strip().splitlines()) == len(HINDI_MIXED_SLIP.strip().splitlines())


def test_normalize_folds_en_dash_so_dose_notation_still_matches() -> None:
    # OCR frequently emits en-dashes where the strip had hyphens.
    ocr_output = normalize_text("Tab. Glycomet 500mg 1–0–1 x 30 days")
    assert DOSE_SLOT_RE.search(ocr_output) is not None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def test_date_is_read_day_first_and_flagged_ambiguous() -> None:
    match = DATE_DMY_RE.search("Date: 03/04/2026")
    assert match is not None
    parsed = parse_date(match)
    assert parsed is not None
    # 3 April, not 4 March. Reading it the other way is a clinical error.
    assert parsed.value == date(2026, 4, 3)
    assert parsed.ambiguous is True
    assert parsed.rule_confidence < 0.8  # we assumed the order; say so honestly


def test_date_with_day_over_twelve_is_unambiguous() -> None:
    match = DATE_DMY_RE.search(LAB_REPORT)
    assert match is not None
    parsed = parse_date(match)
    assert parsed is not None
    assert parsed.value == date(2026, 3, 13)
    assert parsed.ambiguous is False
    assert parsed.rule_confidence > 0.9


def test_all_dates_in_lab_report_are_found_in_order() -> None:
    parsed = [parse_date(m) for m in DATE_DMY_RE.finditer(LAB_REPORT)]
    assert [p.value for p in parsed if p] == [date(2026, 3, 13), date(2026, 3, 14)]


def test_hindi_date_parses_after_normalisation() -> None:
    match = DATE_DMY_RE.search(normalize_text(HINDI_MIXED_SLIP))
    assert match is not None
    parsed = parse_date(match)
    assert parsed is not None and parsed.value == date(2026, 2, 15)


def test_named_month_is_not_ambiguous() -> None:
    match = DATE_DMY_RE.search("Reported on 05-Jan-2026")
    assert match is not None
    parsed = parse_date(match)
    assert parsed is not None
    assert parsed.value == date(2026, 1, 5) and parsed.ambiguous is False


def test_impossible_date_is_dropped_not_repaired() -> None:
    match = DATE_DMY_RE.search("Date: 31/02/2026")
    assert match is not None
    assert parse_date(match) is None


def test_lab_reference_range_is_not_mistaken_for_a_date() -> None:
    # "13.0 - 17.0" must not be swallowed as day-month-year.
    assert DATE_DMY_RE.search("Haemoglobin 11.2 g/dL 13.0 - 17.0") is None


# ---------------------------------------------------------------------------
# Dosing
# ---------------------------------------------------------------------------


def test_dose_notation_across_whole_prescription() -> None:
    schedules = [parse_dose_notation(m) for m in DOSE_SLOT_RE.finditer(PHC_OPD_SLIP)]
    assert [s.slots for s in schedules] == [[1, 0, 1], [1, 0, 0], [0, 0, 1]]
    assert schedules[0].total_per_day == 2


def test_four_slot_notation_includes_bedtime() -> None:
    match = DOSE_SLOT_RE.search("Tab. Augmentin 625  1-1-1-1  x 5 days")
    assert match is not None
    schedule = parse_dose_notation(match)
    assert schedule.slots == [1, 1, 1, 1] and schedule.night == 1


def test_half_tablet_notation() -> None:
    for text, expected in [("1/2-0-1/2", 0.5), ("½-0-½", 0.5), ("0.5-0-0.5", 0.5)]:
        match = DOSE_SLOT_RE.search(f"Tab. Amlong 5mg {text} x 30 days")
        assert match is not None, text
        assert parse_dose_notation(match).morning == expected


def test_fraction_slots_are_not_truncated_to_their_whole_number() -> None:
    # Regression: "1/2" was matching only "1" and leaving "/2" behind, because
    # bare-digit came before the fraction form in the alternation.
    match = DOSE_SLOT_RE.search("Tab. Amlong 5mg 1/2-0-1/2 x 30 days")
    assert match is not None
    schedule = parse_dose_notation(match)
    assert schedule.slots == [0.5, 0.0, 0.5]


def test_dose_pattern_does_not_match_a_date_or_a_phone_number() -> None:
    assert DOSE_SLOT_RE.search("Date: 03-04-2026") is None
    assert DOSE_SLOT_RE.search("Contact 022-2754-1122") is None


def test_hindi_dose_notation_after_normalisation() -> None:
    match = DOSE_SLOT_RE.search(normalize_text("गोली Metformin 500mg १-०-१ x १० दिन"))
    assert match is not None
    assert parse_dose_notation(match).slots == [1, 0, 1]


# ---------------------------------------------------------------------------
# Prescription lines
# ---------------------------------------------------------------------------


def test_drug_lines_capture_brand_name_as_written() -> None:
    names = [m.group("name").strip() for m in DRUG_LINE_RE.finditer(PHC_OPD_SLIP)]
    # Brand names, not generics — resolution happens in terminology/, not here.
    assert names == ["Glycomet", "Telma", "Becosules", "Ascoril"]


def test_drug_line_keeps_the_rest_of_the_line_for_downstream_parsing() -> None:
    match = DRUG_LINE_RE.search(PHC_OPD_SLIP)
    assert match is not None
    rest = match.group("rest")
    assert "500mg" in rest and "1-0-1" in rest and "30 days" in rest


def test_hindi_dosage_form_prefix_is_recognised() -> None:
    match = DRUG_LINE_RE.search(normalize_text(HINDI_MIXED_SLIP))
    assert match is not None
    assert match.group("name").strip() == "Metformin"


def test_strength_units_from_the_slip() -> None:
    found = [(m.group("value"), m.group("unit")) for m in STRENGTH_RE.finditer(PHC_OPD_SLIP)]
    assert ("500", "mg") in found and ("40", "mg") in found and ("5", "ml") in found


def test_frequency_abbreviations() -> None:
    freqs = [m.group("freq").upper() for m in FREQUENCY_RE.finditer(PHC_OPD_SLIP)]
    assert freqs == ["OD", "HS", "TDS"]


@pytest.mark.parametrize("word", ["Odour", "Bdellium", "instat", "SOSpicious"])
def test_frequency_does_not_match_inside_words(word: str) -> None:
    assert FREQUENCY_RE.search(word) is None


def test_duration_english_and_hindi() -> None:
    assert DURATION_RE.search("x 30 days").group("count") == "30"
    assert DURATION_RE.search(normalize_text("x १० दिन")).group("count") == "10"


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------


def test_bp_from_english_slip() -> None:
    match = BP_RE.search(PHC_OPD_SLIP)
    assert match is not None
    assert (match.group("systolic"), match.group("diastolic")) == ("140", "90")


@pytest.mark.parametrize(
    "text",
    ["BP 140/90 mmHg", "Blood Pressure: 140/90", "B.P. - 140/90 mm Hg", "बीपी: 140/90 mmHg"],
)
def test_bp_variants_all_reach_the_same_reading(text: str) -> None:
    # This is the whole pitch: three spellings, one observation.
    match = BP_RE.search(text)
    assert match is not None
    assert (match.group("systolic"), match.group("diastolic")) == ("140", "90")


def test_other_vitals_from_the_slip() -> None:
    assert PULSE_RE.search(PHC_OPD_SLIP).group("value") == "88"
    assert SPO2_RE.search(PHC_OPD_SLIP).group("value") == "97"
    assert TEMPERATURE_RE.search(PHC_OPD_SLIP).group("value") == "98.6"
    assert WEIGHT_RE.search(PHC_OPD_SLIP).group("value") == "78"
    assert HEIGHT_RE.search(PHC_OPD_SLIP).group("value") == "168"


def test_hindi_vitals_after_normalisation() -> None:
    text = normalize_text(HINDI_MIXED_SLIP)
    bp = BP_RE.search(text)
    assert bp is not None and (bp.group("systolic"), bp.group("diastolic")) == ("130", "85")
    assert PULSE_RE.search(text).group("value") == "76"
    assert WEIGHT_RE.search(text).group("value") == "68"


def test_bp_pattern_ignores_an_unlabelled_fraction() -> None:
    # A bare "140/90" with no label is not confidently a BP reading.
    assert BP_RE.search("Ratio 140/90") is None


def test_vital_labels_do_not_match_mid_word() -> None:
    # Regression: bare "T" inside "Weight"/"Height"/"Glycomet" previously matched
    # as a temperature reading (e.g. "500mg" -> "t 50"). These must not fire.
    assert TEMPERATURE_RE.search("Weight 78 kg") is None
    assert TEMPERATURE_RE.search("Height 168 cm") is None
    assert TEMPERATURE_RE.search("Tab. Glycomet 500mg") is None
    assert WEIGHT_RE.search("Height 168 cm") is None


# ---------------------------------------------------------------------------
# Lab rows
# ---------------------------------------------------------------------------


def test_every_analyte_row_in_the_lab_report_is_parsed() -> None:
    rows = {
        m.group("analyte").strip(): (m.group("value"), m.group("unit"))
        for m in LAB_ROW_RE.finditer(LAB_REPORT)
    }
    assert rows["Haemoglobin"] == ("11.2", "g/dL")
    assert rows["Fasting Blood Sugar"] == ("126", "mg/dL")
    assert rows["HbA1c"] == ("7.8", "%")
    assert rows["Total Leucocyte Count"] == ("9800", "cells/cumm")
    assert rows["TSH"] == ("5.9", "mIU/L")


def test_hdl_and_ldl_stay_distinct_rows() -> None:
    # One character apart, clinically opposite. They must never merge.
    rows = {m.group("analyte").strip(): m.group("value") for m in LAB_ROW_RE.finditer(LAB_REPORT)}
    assert rows["HDL Cholesterol"] == "38"
    assert rows["LDL Cholesterol"] == "142"


def test_reference_range_is_captured_not_confused_with_the_result() -> None:
    match = next(m for m in LAB_ROW_RE.finditer(LAB_REPORT) if "Haemoglobin" in m.group("analyte"))
    assert match.group("value") == "11.2"
    assert match.group("range").replace(" ", "") == "13.0-17.0"


def test_lab_row_handles_missing_unit_and_an_operator() -> None:
    text = "Vitamin D                   18\nCRP                         < 5      mg/dL"
    rows = {m.group("analyte").strip(): m for m in LAB_ROW_RE.finditer(text)}
    assert rows["Vitamin D"].group("value") == "18"
    assert rows["Vitamin D"].group("unit") is None
    assert rows["CRP"].group("operator") == "<" and rows["CRP"].group("value") == "5"


def test_lab_row_does_not_match_the_report_header_line() -> None:
    header = "TEST                        RESULT      UNIT            REFERENCE RANGE"
    assert LAB_ROW_RE.search(header) is None


# ---------------------------------------------------------------------------
# Facility
# ---------------------------------------------------------------------------


def test_facility_from_each_document_type() -> None:
    assert "PRIMARY HEALTH CENTRE" in FACILITY_RE.search(PHC_OPD_SLIP).group("facility").upper()
    assert "DIAGNOSTICS" in FACILITY_RE.search(LAB_REPORT).group("facility").upper()
    assert "स्वास्थ्य केंद्र" in FACILITY_RE.search(HINDI_MIXED_SLIP).group("facility")