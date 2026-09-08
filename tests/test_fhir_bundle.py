"""Tests for whole-DocumentBundle FHIR assembly.

Fixtures are hand-built FHIR fragments shaped exactly like what fhir/builders.py
produces per test_pipeline.py (DiagnosticReport for labs, Bundle-of-MedicationRequest
for prescriptions) — not invented shapes convenient for the assembler.
"""
from __future__ import annotations

from sanjeevani_ml.fhir.bundle import build_document_bundle_fhir
from sanjeevani_ml.schemas import (
    AbnormalValue,
    DocumentBundle,
    InteractionAlert,
    TimelineEntry,
)


def _fresh_lab_fhir() -> dict:
    """A new DiagnosticReport dict every call.

    `build_document_bundle_fhir` mutates resources in place to attach abnormal-
    value markers — correct behaviour for real use, but it means tests sharing
    one dict object would leak an extension from one test into the next. A
    factory, not a module-level constant, keeps each test's fixture isolated.
    """
    return {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",
        "contained": [
            {
                "resourceType": "Observation",
                "id": "obs-1",
                "code": {"text": "Fasting Blood Sugar"},
                "valueQuantity": {"value": 142, "unit": "mg/dL"},
            },
        ],
    }


def _fresh_prescription_fhir() -> dict:
    """A new Bundle-of-MedicationRequest dict every call — see `_fresh_lab_fhir`."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {
                "resourceType": "MedicationRequest", "id": "med-1",
                "medicationCodeableConcept": {"text": "warfarin"},
            }},
            {"resource": {
                "resourceType": "MedicationRequest", "id": "med-2",
                "medicationCodeableConcept": {"text": "aspirin"},
            }},
        ],
    }


def _lab_entry(document_id="lab-1", fhir=None) -> TimelineEntry:
    return TimelineEntry(
        date="2026-03-14", kind="lab", title="Lab Report", document_id=document_id,
        fhir=fhir if fhir is not None else _fresh_lab_fhir(),
    )


def _rx_entry(document_id="rx-1", fhir=None) -> TimelineEntry:
    return TimelineEntry(
        date="2026-04-02", kind="prescription", title="Prescription",
        document_id=document_id, fhir=fhir if fhir is not None else _fresh_prescription_fhir(),
    )


class TestBundleShape:
    def test_produces_a_collection_bundle(self):
        bundle = DocumentBundle(session_id="s1", timeline=[_lab_entry()])
        result = build_document_bundle_fhir(bundle)
        assert result["resourceType"] == "Bundle"
        assert result["type"] == "collection"

    def test_is_tagged_as_an_ml_draft_never_final(self):
        bundle = DocumentBundle(session_id="s1", timeline=[_lab_entry()])
        result = build_document_bundle_fhir(bundle)
        tags = result["meta"]["tag"]
        assert any("draft" in t["code"] for t in tags)

    def test_an_empty_document_bundle_produces_a_valid_empty_bundle(self):
        bundle = DocumentBundle(session_id="s1", timeline=[])
        result = build_document_bundle_fhir(bundle)
        assert result["resourceType"] == "Bundle"
        assert result["entry"] == []
        assert result["total"] == 0


class TestResourceCollection:
    def test_a_single_document_resource_is_included_directly(self):
        bundle = DocumentBundle(session_id="s1", timeline=[_lab_entry()])
        result = build_document_bundle_fhir(bundle)
        resource_types = [e["resource"]["resourceType"] for e in result["entry"]]
        assert "DiagnosticReport" in resource_types

    def test_a_nested_bundle_is_flattened_not_nested_inside_a_bundle(self):
        """A prescription's own Bundle-of-MedicationRequests is unwrapped, not nested."""
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        result = build_document_bundle_fhir(bundle)
        resource_types = [e["resource"]["resourceType"] for e in result["entry"]]
        assert resource_types.count("Bundle") == 0
        assert resource_types.count("MedicationRequest") == 2

    def test_multiple_documents_all_appear(self):
        bundle = DocumentBundle(
            session_id="s1", timeline=[_lab_entry("lab-1"), _rx_entry("rx-1")],
        )
        result = build_document_bundle_fhir(bundle)
        resource_types = {e["resource"]["resourceType"] for e in result["entry"]}
        assert "DiagnosticReport" in resource_types
        assert "MedicationRequest" in resource_types


class TestUnstructuredDocumentPlaceholder:
    def test_a_document_with_no_fhir_becomes_a_placeholder_not_dropped(self):
        """A document extraction couldn't structure must still appear in the record."""
        entry = TimelineEntry(
            date="2026-05-01", kind="note", title="Illegible handwritten note",
            document_id="note-1", fhir={},
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        result = build_document_bundle_fhir(bundle)
        resource_types = [e["resource"]["resourceType"] for e in result["entry"]]
        assert "DocumentReference" in resource_types

    def test_the_placeholder_names_the_original_document(self):
        entry = TimelineEntry(
            date=None, kind="note", title="Illegible handwritten note",
            document_id="note-99", fhir={},
        )
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        result = build_document_bundle_fhir(bundle)
        placeholder = next(
            e["resource"] for e in result["entry"]
            if e["resource"]["resourceType"] == "DocumentReference"
        )
        ext = placeholder["extension"][0]
        assert ext["valueString"] == "note-99"

    def test_the_placeholder_is_never_claimed_as_structured_data(self):
        entry = TimelineEntry(date=None, kind="note", title="Unreadable", fhir={})
        bundle = DocumentBundle(session_id="s1", timeline=[entry])
        result = build_document_bundle_fhir(bundle)
        placeholder = next(
            e["resource"] for e in result["entry"]
            if e["resource"]["resourceType"] == "DocumentReference"
        )
        assert placeholder["docStatus"] == "preliminary"
        assert "could not be automatically structured" in placeholder["description"]


class TestAbnormalValueMarkers:
    def test_a_flagged_lab_gets_an_abnormal_extension_on_its_observation(self):
        """The Observation lives inside DiagnosticReport.contained, not top-level —
        matching fhir/builders.py's real shape (see test_pipeline.py's own
        `len(result.fhir['contained']) >= 3` assertion)."""
        bundle = DocumentBundle(
            session_id="s1",
            timeline=[_lab_entry()],
            abnormal_values=[
                AbnormalValue(
                    analyte="Fasting Blood Sugar", value=142, unit="mg/dL",
                    reference_range="70.0-100.0", direction="high",
                    source_document_id="lab-1",
                )
            ],
        )
        result = build_document_bundle_fhir(bundle)
        report = next(
            e["resource"] for e in result["entry"]
            if e["resource"]["resourceType"] == "DiagnosticReport"
        )
        obs = report["contained"][0]
        assert "extension" in obs
        assert obs["extension"][0]["url"].endswith("abnormal-value")

    def test_a_normal_lab_gets_no_abnormal_extension(self):
        bundle = DocumentBundle(session_id="s1", timeline=[_lab_entry()], abnormal_values=[])
        result = build_document_bundle_fhir(bundle)
        report = next(
            e["resource"] for e in result["entry"]
            if e["resource"]["resourceType"] == "DiagnosticReport"
        )
        obs = report["contained"][0]
        assert "extension" not in obs or not obs["extension"]


class TestInteractionIssues:
    def test_an_interaction_alert_becomes_a_detected_issue(self):
        bundle = DocumentBundle(
            session_id="s1",
            timeline=[_rx_entry()],
            interactions=[
                InteractionAlert(
                    severity="major", drug_a="warfarin", drug_b="aspirin",
                    description="Increased bleeding risk",
                )
            ],
        )
        result = build_document_bundle_fhir(bundle)
        resource_types = [e["resource"]["resourceType"] for e in result["entry"]]
        assert "DetectedIssue" in resource_types

    def test_the_detected_issue_references_both_implicated_medications(self):
        bundle = DocumentBundle(
            session_id="s1",
            timeline=[_rx_entry()],
            interactions=[
                InteractionAlert(
                    severity="major", drug_a="warfarin", drug_b="aspirin",
                    description="Increased bleeding risk",
                )
            ],
        )
        result = build_document_bundle_fhir(bundle)
        issue = next(
            e["resource"] for e in result["entry"]
            if e["resource"]["resourceType"] == "DetectedIssue"
        )
        assert len(issue["implicated"]) == 2

    def test_an_interaction_naming_an_untraceable_drug_still_appears(self):
        """The warning is never dropped just because we can't trace one side of it."""
        bundle = DocumentBundle(
            session_id="s1",
            timeline=[_rx_entry()],
            interactions=[
                InteractionAlert(
                    severity="major", drug_a="warfarin", drug_b="some_unlisted_drug",
                    description="Unknown interaction",
                )
            ],
        )
        result = build_document_bundle_fhir(bundle)
        issue = next(
            e["resource"] for e in result["entry"]
            if e["resource"]["resourceType"] == "DetectedIssue"
        )
        assert len(issue["implicated"]) == 1  # only warfarin traced

    def test_severity_maps_to_a_fhir_severity_code(self):
        bundle = DocumentBundle(
            session_id="s1", timeline=[],
            interactions=[
                InteractionAlert(severity="major", drug_a="a", drug_b="b", description="x")
            ],
        )
        result = build_document_bundle_fhir(bundle)
        issue = result["entry"][0]["resource"]
        assert issue["severity"] == "high"
