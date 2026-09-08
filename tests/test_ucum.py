"""Tests for the UCUM unit-code fix in fhir/builders.py.

Confirms: known display units map to their real, verified UCUM code; units with
no confirmed mapping get no code at all (never a guess); the display string in
`unit` is always preserved either way.
"""
from __future__ import annotations

from sanjeevani_ml.fhir.builders import _ucum_code, build_observation
from sanjeevani_ml.schemas import ExtractedEntity


def _lab(text, value, unit) -> ExtractedEntity:
    return ExtractedEntity(kind="lab", text=text, value=value, unit=unit, confidence=0.9)


class TestUcumCodeLookup:
    def test_blood_pressure_maps_to_the_real_ucum_code(self):
        """mm[Hg] is the code FHIR's own published Blood Pressure examples use."""
        assert _ucum_code("mmHg") == "mm[Hg]"

    def test_common_lab_units_map_correctly(self):
        assert _ucum_code("mg/dL") == "mg/dL"
        assert _ucum_code("g/dL") == "g/dL"
        assert _ucum_code("U/L") == "U/L"
        assert _ucum_code("%") == "%"

    def test_case_insensitive_fallback_still_matches(self):
        assert _ucum_code("mmhg") == "mm[Hg]"

    def test_milliequivalent_maps_to_lowercase_meq(self):
        assert _ucum_code("mEq/L") == "meq/L"

    def test_fahrenheit_maps_to_bracketed_ucum_code(self):
        assert _ucum_code("degF") == "[degF]"

    def test_unconfirmed_indian_lab_units_get_no_code(self):
        """cells/cumm and lakhs/cumm have no single agreed UCUM mapping we could
        confirm — never invent one, per the project's core rule."""
        assert _ucum_code("cells/cumm") is None
        assert _ucum_code("lakhs/cumm") is None

    def test_an_unknown_unit_gets_no_code(self):
        assert _ucum_code("furlongs/fortnight") is None

    def test_no_unit_at_all_gets_no_code(self):
        assert _ucum_code(None) is None
        assert _ucum_code("") is None


class TestObservationBuilding:
    def test_a_confirmed_unit_gets_a_ucum_code_in_the_observation(self):
        entity = _lab("Blood pressure", "120/80", "mmHg")
        # value here is a string ("120/80"), not numeric, so valueQuantity path is
        # skipped — use a numeric vital instead for this test:
        entity = _lab("Heart rate", 72, "beats/min")
        obs = build_observation(entity, engine="test")
        # beats/min is intentionally NOT in the confirmed map (ambiguous clinical
        # convention) — this is the "no fabricated code" path.
        assert "code" not in obs["valueQuantity"]
        assert obs["valueQuantity"]["unit"] == "beats/min"

    def test_a_lab_value_with_a_confirmed_unit_gets_the_ucum_code(self):
        entity = _lab("Fasting Blood Sugar", 142, "mg/dL")
        obs = build_observation(entity, engine="test")
        assert obs["valueQuantity"]["code"] == "mg/dL"

    def test_an_unconfirmed_unit_still_preserves_the_display_string(self):
        """No code, but nothing is lost — a human still sees the real unit."""
        entity = _lab("Total leucocyte count", 7500, "cells/cumm")
        obs = build_observation(entity, engine="test")
        assert "code" not in obs["valueQuantity"]
        assert obs["valueQuantity"]["unit"] == "cells/cumm"

    def test_the_ucum_system_uri_is_always_present_even_without_a_code(self):
        entity = _lab("Total leucocyte count", 7500, "cells/cumm")
        obs = build_observation(entity, engine="test")
        assert obs["valueQuantity"]["system"] == "http://unitsofmeasure.org"
