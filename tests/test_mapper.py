"""Tests for terminology/mapper.py, run against the team's real curated CSVs.

These use the actual ml/data/*.csv content (mirrored here), not synthetic
fixtures, since a wrong code against real data is exactly the failure mode
this module exists to prevent.
"""

from __future__ import annotations

from sanjeevani_ml.extraction.structurer import structure
from sanjeevani_ml.schemas import ExtractedEntity, OcrResult
from sanjeevani_ml.terminology.mapper import code_result, generic_name, normalize


def _ocr(text: str) -> OcrResult:
    return OcrResult(text=text, mean_confidence=0.92, engine="test-fixture")


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_folds_case_and_punctuation() -> None:
    assert normalize("Warfarin") == normalize("WARFARIN ") == normalize("warfarin")


def test_normalize_collapses_punctuation_to_spaces() -> None:
    assert normalize("Type 2 Diabetes Mellitus") == normalize("Type-2, Diabetes  Mellitus!")


# ---------------------------------------------------------------------------
# generic_name()
# ---------------------------------------------------------------------------


def test_generic_name_resolves_common_indian_brands() -> None:
    assert generic_name("Glycomet") == "metformin"
    assert generic_name("Amlong") == "amlodipine"
    assert generic_name("Ecosprin") == "aspirin"
    assert generic_name("Crocin") == "paracetamol"


def test_generic_name_is_case_insensitive() -> None:
    assert generic_name("glycomet") == "metformin"
    assert generic_name("GLYCOMET") == "metformin"


def test_generic_name_accepts_the_generic_itself() -> None:
    assert generic_name("metformin") == "metformin"
    assert generic_name("Metformin") == "metformin"


def test_generic_name_returns_none_for_a_drug_outside_the_table() -> None:
    # None is a correct, honest answer — never guess a generic for an unknown brand.
    assert generic_name("SomeCompletelyMadeUpBrandXYZ") is None


def test_generic_name_returns_none_for_empty_input() -> None:
    assert generic_name("") is None
    assert generic_name("   ") is None


# ---------------------------------------------------------------------------
# code_result() — labs
# ---------------------------------------------------------------------------


def test_lab_entity_gets_the_correct_loinc_code() -> None:
    result = code_result(structure(_ocr("Fasting Blood Sugar    126   mg/dL   (70-100)")))
    fbs = result.entities[0]
    assert fbs.code is not None
    assert fbs.code.system == "LOINC"
    assert fbs.code.code == "1558-6"


def test_lab_entity_matches_via_a_synonym() -> None:
    # "FBS" is a synonym for "Fasting blood sugar" in loinc_seed.csv, not the term itself.
    result = code_result(structure(_ocr("FBS    126   mg/dL")))
    fbs = next(e for e in result.entities if e.kind == "lab")
    assert fbs.code is not None and fbs.code.code == "1558-6"


def test_hba1c_maps_correctly_and_stays_distinct_from_fbs() -> None:
    result = code_result(structure(_ocr("HbA1c    7.8   %   (4.0-5.6)")))
    hba1c = result.entities[0]
    assert hba1c.code is not None
    assert hba1c.code.code == "4548-4"
    assert hba1c.code.code != "1558-6"


def test_lab_entity_with_no_curated_match_keeps_code_none() -> None:
    # A lab test genuinely outside our ~4-row seed table for this test run.
    result = code_result(structure(_ocr("Serum Ferritin    45   ng/mL")))
    ferritin = result.entities[0]
    assert ferritin.kind == "lab"
    assert ferritin.code is None
    assert ferritin.text == "Serum Ferritin"  # original text untouched, not guessed at


# ---------------------------------------------------------------------------
# code_result() — diagnoses
# ---------------------------------------------------------------------------


def test_diagnosis_maps_to_snomed_via_synonym() -> None:
    # "Sugar" is a colloquial synonym for Type 2 diabetes in snomed_seed.csv.
    result = code_result(structure(_ocr("Dx: Sugar")))
    diagnosis = result.entities[0]
    assert diagnosis.code is not None
    assert diagnosis.code.system == "SNOMED"
    assert diagnosis.code.code == "44054006"


def test_two_diagnoses_on_one_line_each_get_their_own_code() -> None:
    result = code_result(structure(_ocr("Dx: Type 2 Diabetes Mellitus, Hypertension")))
    diagnoses = [e for e in result.entities if e.kind == "diagnosis"]
    assert len(diagnoses) == 2
    codes = {d.code.code for d in diagnoses if d.code}
    assert codes == {"44054006", "59621000"}


def test_uncoded_diagnosis_keeps_code_none_not_a_guess() -> None:
    result = code_result(structure(_ocr("Dx: Some Extremely Rare Undocumented Syndrome")))
    diagnosis = result.entities[0]
    assert diagnosis.code is None


# ---------------------------------------------------------------------------
# code_result() — medications
# ---------------------------------------------------------------------------


def test_medication_entity_gets_rxnorm_code_from_brand_name() -> None:
    result = code_result(structure(_ocr("Tab. Glycomet 500mg 1-0-1 x 30 days")))
    med = result.entities[0]
    assert med.code is not None
    assert med.code.system == "RxNorm"
    assert med.code.code == "6809"  # metformin's RxNorm code in drugs_seed.csv


def test_medication_annotation_does_not_break_name_re_extraction() -> None:
    # entity.text includes the plain-language dose phrase appended in brackets;
    # coding must still find the drug name buried inside it.
    result = code_result(structure(_ocr("Tab. Amlong 5mg 1-0-0 x 30 days")))
    med = result.entities[0]
    assert "[1 tablet in the morning" in med.text  # annotation present
    assert med.code is not None and med.code.code == "17767"  # amlodipine


def test_unknown_brand_medication_keeps_code_none() -> None:
    result = code_result(structure(_ocr("Tab. TotallyFictionalBrandXQZ 10mg 1-0-0 x 5 days")))
    med = result.entities[0]
    assert med.code is None


# ---------------------------------------------------------------------------
# code_result() — never invents, never touches vitals/dates
# ---------------------------------------------------------------------------


def test_vitals_and_dates_are_left_uncoded() -> None:
    text = "BP 140/90 mmHg   Pulse 88 bpm\nDate: 03/04/2026"
    result = code_result(structure(_ocr(text)))
    non_terminology_kinds = {e.kind for e in result.entities if e.code is not None}
    assert "vital" not in non_terminology_kinds
    assert "date" not in non_terminology_kinds


def test_code_result_never_mutates_the_input_result_in_place() -> None:
    original = structure(_ocr("Fasting Blood Sugar    126   mg/dL"))
    original_code = original.entities[0].code
    code_result(original)
    assert original.entities[0].code == original_code  # still None — input untouched


def test_full_prescription_round_trip_codes_every_matchable_drug() -> None:
    text = """
Rx
Tab. Glycomet 500mg 1-0-1 x 30 days
Tab. Amlong 5mg 1-0-0 x 30 days
Cap. Omez 20mg 1-0-0 x 14 days
"""
    result = code_result(structure(_ocr(text)))
    meds: list[ExtractedEntity] = [e for e in result.entities if e.kind == "medication"]
    assert len(meds) == 3
    assert all(m.code is not None and m.code.system == "RxNorm" for m in meds)

# --- terminology table hygiene -------------------------------------------------
# Offline on purpose: these run in CI without network. The network check (does this
# RxCUI exist, and is it TTY=IN) belongs in scripts/fetch_rxnorm_codes.py, which is
# where codes enter the table.

def _drug_rows():
    import csv
    from pathlib import Path

    from sanjeevani_ml.config import settings

    with (Path(settings.data_dir) / "drugs_seed.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


#: RxNorm names a few ingredients with a qualifier, and these are TTY=IN - verified
#: against RxNav, not assumed. The shape check below cannot tell them from a salt
#: without a network call, so they are named here instead of the data being bent to
#: fit the test.
_QUALIFIED_INGREDIENTS = {
    "insulin aspart": "insulin aspart, human",
    "ispaghula": "ispaghula extract",
}


def test_drug_codes_are_ingredients_not_salts():
    """A script reading "Metronidazole 400mg" does not say *benzoate*.

    RxNorm PIN concepts name a specific salt or ester. Coding a plain generic mention
    as one asserts a fact the paper never contained - the same sin as inventing a code,
    and it silently changes which product a pharmacist dispenses. Five rows entered the
    table this way; the shape that catches it is the display carrying words the generic
    does not.
    """
    for row in _drug_rows():
        generic, display = row["generic"].strip().lower(), row["display"].strip().lower()
        if _QUALIFIED_INGREDIENTS.get(generic) == display:
            continue
        # A different word entirely is a synonym and fine - RxNorm's official name for
        # paracetamol is "acetaminophen". What is NOT fine is the generic plus extra
        # qualifying words, which is how a salt looks.
        assert not (display.startswith(generic) and len(display) > len(generic)), (
            f"{row['generic']}: display is {row['display']!r} - more specific than the "
            f"generic. Use the IN (ingredient) RxCUI, not the PIN (salt/ester)."
        )


def test_no_duplicate_drug_codes_or_generics():
    """Two rows for one drug means one of them silently never matches."""
    rows = _drug_rows()
    for field in ("generic", "code"):
        values = [r[field].strip().lower() for r in rows]
        dupes = {v for v in values if values.count(v) > 1}
        assert not dupes, f"duplicate {field}: {sorted(dupes)}"


def test_every_drug_row_is_complete():
    """A blank code is a term the pipeline reads and cannot code."""
    for row in _drug_rows():
        for field in ("generic", "system", "code", "display", "class"):
            assert row.get(field, "").strip(), f"{row.get('generic')}: {field} is empty"
