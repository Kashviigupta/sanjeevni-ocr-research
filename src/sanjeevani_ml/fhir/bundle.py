"""Package a whole DocumentBundle into one FHIR R4 Bundle.

Each `TimelineEntry.fhir` already holds a per-document resource, built earlier by
`terminology.mapper.code_result` -> `fhir.builders.build`. This module does not
rebuild those — it collects them, in timeline order, into one `Bundle` resource
ready to hand to the backend for the HIS / ABHA PHR push, and adds the two things
a single document's FHIR can't express on its own: an abnormal-value marker that
spans "this Observation vs this document", and drug-interaction warnings that
span multiple MedicationRequests possibly written by different prescribers.

Nothing here talks to a server. This is pure data assembly — Ronit's backend
decides when and how a Bundle is actually pushed.
"""
from __future__ import annotations

from typing import Any

from sanjeevani_ml.schemas import AbnormalValue, DocumentBundle, InteractionAlert, TimelineEntry

#: Marks every bundle this module produces as a draft artefact of the ML service,
#: never a physician-confirmed record. Same spirit as `status: "preliminary"` on
#: individual resources in fhir/builders.py — a downstream system must not treat
#: this as final just because it is valid FHIR.
_DRAFT_TAG = {
    "system": "https://sanjeevani.aiia.gov.in/fhir/tags",
    "code": "ml-draft-bundle",
    "display": "Assembled by the ML service; not yet reviewed by a physician",
}

#: Extension URLs. Namespaced under our own domain per FHIR convention for
#: locally-defined extensions — these are not part of the base FHIR spec.
_ABNORMAL_EXTENSION_URL = "https://sanjeevani.aiia.gov.in/fhir/extension/abnormal-value"
_SOURCE_DOCUMENT_EXTENSION_URL = "https://sanjeevani.aiia.gov.in/fhir/extension/source-document-id"


def _document_reference_placeholder(entry: TimelineEntry) -> dict[str, Any]:
    """A stand-in resource for a document extraction could not structure.

    Dropping a document the patient handed over is exactly the failure the
    `unreadable` list exists to prevent for OCR failures; this is the same
    principle one stage later — a document that read fine but produced no
    FHIR resource (e.g. an unrecognised document type) must still appear in
    the bundle, not vanish from the patient's record.
    """
    return {
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "preliminary",
        "type": {"text": entry.title or "Unstructured document"},
        "date": entry.date,
        "description": (
            "This document could not be automatically structured. "
            "The original scan is available for manual review."
        ),
        "extension": [
            {"url": _SOURCE_DOCUMENT_EXTENSION_URL, "valueString": entry.document_id}
        ] if entry.document_id else [],
    }


def _entry_resources(entry: TimelineEntry) -> list[dict[str, Any]]:
    """The resource(s) one timeline entry contributes to the bundle.

    A document's own `fhir` dict may itself be a `Bundle` (e.g. a multi-drug
    prescription, per fhir/builders.py) rather than a single resource — in
    that case we lift its entries out rather than nesting a Bundle inside a
    Bundle, which is legal FHIR but awkward for a consumer to walk.
    """
    fhir = entry.fhir
    if not fhir:
        return [_document_reference_placeholder(entry)]
    if fhir.get("resourceType") == "Bundle":
        return [e["resource"] for e in fhir.get("entry", []) if "resource" in e]
    return [fhir]


def _attach_abnormal_markers(
    resources: list[dict[str, Any]], abnormal_values: list[AbnormalValue]
) -> None:
    """Mark the Observation resources that correspond to a flagged lab value, in place.

    Matched on analyte name (case-insensitive) plus source document id — the
    same pairing `timeline.abnormal.flag_abnormal_values` used to produce the
    AbnormalValue in the first place. An Observation with no matching flag is
    left untouched; this never invents a marker for a resource we can't trace
    back to a specific out-of-range result.

    A lab result's Observation is usually NOT a top-level bundle entry — per
    fhir/builders.py, a lab document's FHIR is a DiagnosticReport with its
    Observations nested in `contained`. We walk both top-level resources and
    everything in their `contained` array, since mutating either mutates the
    same dict this function was handed (contained items are ordinary nested
    dicts, not copies).
    """
    index: dict[tuple[str, str | None], AbnormalValue] = {
        (a.analyte.lower(), a.source_document_id): a for a in abnormal_values
    }

    def observations(resource: dict[str, Any]):
        if resource.get("resourceType") == "Observation":
            yield resource
        for contained in resource.get("contained", []):
            yield from observations(contained)

    for top_level in resources:
        for observation in observations(top_level):
            analyte = (observation.get("code", {}).get("text") or "").lower()
            for (key_analyte, _doc_id), abnormal in index.items():
                if key_analyte and key_analyte == analyte:
                    extensions = observation.setdefault("extension", [])
                    extensions.append({
                        "url": _ABNORMAL_EXTENSION_URL,
                        "extension": [
                            {"url": "direction", "valueCode": abnormal.direction},
                            {"url": "referenceRange", "valueString": abnormal.reference_range},
                        ],
                    })
                    break


def _detected_issue(alert: InteractionAlert, medication_resource_ids: dict[str, str]) -> dict[str, Any]:
    """One DetectedIssue resource for one drug-drug interaction alert.

    `DetectedIssue` is the FHIR-standard resource for exactly this: a clinical
    concern that spans more than one other resource. References are included
    only for drugs we can trace to a specific MedicationRequest in this
    bundle — an alert about a drug we can't trace still appears (the warning
    itself is never dropped), just without a `implicated` reference for that
    side.
    """
    implicated = [
        {"reference": f"MedicationRequest/{medication_resource_ids[name]}"}
        for name in (alert.drug_a, alert.drug_b)
        if name in medication_resource_ids
    ]
    return {
        "resourceType": "DetectedIssue",
        "status": "preliminary",
        "severity": {
            "major": "high", "contraindicated": "high",
            "moderate": "moderate", "minor": "low",
        }.get(alert.severity, "moderate"),
        "code": {"text": "Drug-drug interaction"},
        "detail": alert.description,
        "implicated": implicated,
        "extension": [{"url": _SOURCE_DOCUMENT_EXTENSION_URL, "valueString": alert.source}],
    }


def _medication_resource_ids(resources: list[dict[str, Any]]) -> dict[str, str]:
    """Generic-or-brand drug name -> this bundle's resource id, for MedicationRequests.

    Best-effort only: matched on whatever text FHIR builders put in
    `medicationCodeableConcept.text`. An interaction alert naming a drug we
    can't find here still appears in the bundle (see `_detected_issue`) —
    this index only improves traceability when it can, it never gates
    whether the warning itself is included.
    """
    index: dict[str, str] = {}
    for resource in resources:
        if resource.get("resourceType") != "MedicationRequest":
            continue
        text = resource.get("medicationCodeableConcept", {}).get("text", "")
        resource_id = resource.get("id")
        if text and resource_id:
            index[text.lower()] = resource_id
    return index


def build_document_bundle_fhir(bundle: DocumentBundle) -> dict[str, Any]:
    """Assemble one patient's whole document history into a single FHIR Bundle.

    `type: "collection"` — this bundle is a set of independent resources about
    one patient's records, not an atomic set of server operations (`transaction`)
    and not a signed clinical document (`document`). Ronit's backend chooses the
    push mechanics; this function only describes what the patient's paper
    contained.

    Never claims physician review: every bundle carries `_DRAFT_TAG`, matching
    the `status: "preliminary"` discipline already used on individual resources.

    Args:
        bundle: The output of `timeline.builder.build_timeline`, already dated,
            de-duplicated, ordered, and abnormal-value-flagged.

    Returns:
        A FHIR R4 `Bundle` resource. Never raises: an empty `DocumentBundle`
        produces a valid, empty-of-entries Bundle rather than an error, per
        the platform's degrade-never-fail principle.
    """
    all_resources: list[dict[str, Any]] = []
    for entry in bundle.timeline:
        all_resources.extend(_entry_resources(entry))

    _attach_abnormal_markers(all_resources, bundle.abnormal_values)

    medication_ids = _medication_resource_ids(all_resources)
    issue_resources = [
        _detected_issue(alert, medication_ids) for alert in bundle.interactions
    ]

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "meta": {"tag": [_DRAFT_TAG]},
        "entry": [{"resource": r} for r in [*all_resources, *issue_resources]],
        # Counts only, per the platform's no-PHI-in-anything-that-might-get-
        # logged discipline — this total is safe to log, the resources are not.
        "total": len(all_resources) + len(issue_resources),
    }
