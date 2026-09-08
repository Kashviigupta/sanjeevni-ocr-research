"""Coded entities to FHIR R4 resources.

FHIR R4 is the platform's *internal canonical model*, not an export format bolted on at
the end (spec principle 3). Whatever this module emits is what a patient's record
actually is, so the resources must be valid enough that a real FHIR server would accept
them.

Validate anything you are unsure of against https://hl7.org/fhir/R4/ — and remember that
every resource we emit carries `status: preliminary` until the patient verifies it.
"""
from __future__ import annotations

from typing import Any

from ..schemas import ExtractedEntity, ExtractionResult, TerminologyCode

#: Canonical system URIs. Getting these wrong makes the data non-interoperable in a way
#: that is invisible until someone else's system rejects it.
SYSTEM_URI = {
    "LOINC": "http://loinc.org",
    "SNOMED": "http://snomed.info/sct",
    "ICD-11": "http://id.who.int/icd/release/11/mms",
    "RxNorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "NAMASTE": "https://namaste.ayush.gov.in/terminology",
}

#: Display units this pipeline actually produces, mapped to their real UCUM code.
#: Confirmed against FHIR's own published examples (e.g. mm[Hg] for blood pressure,
#: per HL7's US Core Blood Pressure profile) and the UCUM specification's standard
#: unit atoms — not guessed.
#:
#: Two units seen in ml/data/loinc_seed.csv are deliberately absent: "cells/cumm"
#: and "lakhs/cumm" are Indian-lab reporting conventions with no single agreed
#: UCUM equivalent confirmed with confidence (candidates like "/uL" or "10*5/uL"
#: are syntactically legal UCUM but encode a real clinical assumption about what
#: the raw number means — thousands? absolute count? — that this module has no
#: basis to make on its own). Per the never-invent-a-code rule, an unmapped unit
#: gets no `code` at all rather than a guess; the display string stays in `unit`
#: either way, so nothing is lost, only the coded form.
UCUM_UNIT_MAP: dict[str, str] = {
    "mg/dl": "mg/dL", "mg/dL": "mg/dL",
    "%": "%",
    "g/dl": "g/dL", "g/dL": "g/dL",
    "u/l": "U/L", "U/L": "U/L",
    "ng/ml": "ng/mL", "ng/mL": "ng/mL",
    "pg/ml": "pg/mL", "pg/mL": "pg/mL",
    "mg/l": "mg/L", "mg/L": "mg/L",
    "kg": "kg",
    "mm/hr": "mm/h",
    "meq/l": "meq/L", "mEq/L": "meq/L",
    "mmhg": "mm[Hg]", "mmHg": "mm[Hg]",
    "degf": "[degF]", "degF": "[degF]",
    "miu/l": "m[IU]/L", "mIU/L": "m[IU]/L",
}


def _ucum_code(unit: str | None) -> str | None:
    """The real UCUM code for a display unit, or None if we don't have a confirmed one.

    None is a correct, honest answer here — same principle as terminology.mapper
    never inventing a LOINC/SNOMED code below its match threshold. A FHIR Quantity's
    `code` element is optional; omitting it is valid, a wrong code is not.
    """
    if not unit:
        return None
    return UCUM_UNIT_MAP.get(unit) or UCUM_UNIT_MAP.get(unit.lower())


def _codeable(code: TerminologyCode | None, text: str) -> dict[str, Any]:
    """A CodeableConcept. `text` is always populated, coding only when we have a code.

    FHIR explicitly allows a CodeableConcept with text and no coding — that is the
    correct representation of "we read this but could not code it", and far better than
    omitting the finding.
    """
    concept: dict[str, Any] = {"text": text}
    if code:
        concept["coding"] = [{
            "system": SYSTEM_URI.get(code.system, code.system),
            "code": code.code,
            "display": code.display,
        }]
    return concept


def _provenance_extension(engine: str, confidence: float) -> list[dict[str, Any]]:
    """Machine provenance rides on the resource itself.

    A clinician looking at a number must always be able to ask "where did this come
    from?" and get "OCR, 0.87 confidence, patient-verified" rather than silence.
    """
    return [
        {"url": "https://sanjeevani.health/fhir/StructureDefinition/extraction-engine",
         "valueString": engine},
        {"url": "https://sanjeevani.health/fhir/StructureDefinition/extraction-confidence",
         "valueDecimal": round(confidence, 3)},
    ]


def build_observation(entity: ExtractedEntity, *, engine: str,
                      effective: str | None = None) -> dict[str, Any]:
    """One lab value or vital sign.

    `status: preliminary` is deliberate — it becomes `final` only after the patient
    verifies the draft in the app. Machine output is never final clinical truth.
    """
    obs: dict[str, Any] = {
        "resourceType": "Observation",
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory" if entity.kind == "lab" else "vital-signs",
            }],
        }],
        "code": _codeable(entity.code, entity.text),
        "extension": _provenance_extension(engine, entity.confidence),
    }
    if effective:
        obs["effectiveDateTime"] = effective

    if isinstance(entity.value, (int, float)):
        quantity: dict[str, Any] = {
            "value": entity.value,
            "unit": entity.unit or "",
            "system": "http://unitsofmeasure.org",   # UCUM
        }
        ucum = _ucum_code(entity.unit)
        if ucum is not None:
            quantity["code"] = ucum
        # No `code` key at all when we don't have a confirmed UCUM mapping — see
        # UCUM_UNIT_MAP's docstring. `unit` (the display string) is still set, so
        # a human reading the resource loses nothing; only the machine-coded form
        # is honestly absent instead of silently wrong.
        obs["valueQuantity"] = quantity
    elif entity.value is not None:
        obs["valueString"] = str(entity.value)

    if entity.reference_range:
        low, _, high = entity.reference_range.partition("-")
        try:
            obs["referenceRange"] = [{
                "low": {"value": float(low), "unit": entity.unit or ""},
                "high": {"value": float(high), "unit": entity.unit or ""},
            }]
        except ValueError:
            obs["referenceRange"] = [{"text": entity.reference_range}]

    return obs


def build_medication_request(entity: ExtractedEntity, *, engine: str,
                             authored: str | None = None) -> dict[str, Any]:
    """One prescribed drug.

    `intent: order` with `status: draft` — a scanned prescription is evidence that a
    drug was prescribed, not an instruction our system is issuing.
    """
    req: dict[str, Any] = {
        "resourceType": "MedicationRequest",
        "status": "draft",
        "intent": "order",
        "medicationCodeableConcept": _codeable(entity.code, str(entity.value or entity.text)),
        "dosageInstruction": [{"text": entity.text}],
        "extension": _provenance_extension(engine, entity.confidence),
    }
    if authored:
        req["authoredOn"] = authored
    return req


def build_condition(entity: ExtractedEntity, *, engine: str,
                    onset: str | None = None) -> dict[str, Any]:
    """One diagnosis. `verificationStatus: provisional` until a human confirms it."""
    cond: dict[str, Any] = {
        "resourceType": "Condition",
        "clinicalStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": "active"}]},
        "verificationStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "code": "provisional"}]},
        "code": _codeable(entity.code, entity.text),
        "extension": _provenance_extension(engine, entity.confidence),
    }
    if onset:
        cond["onsetDateTime"] = onset
    return cond


def build_diagnostic_report(result: ExtractionResult,
                            observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a lab report's observations in the report that contained them.

    `contained` keeps the bundle self-sufficient: the backend encrypts one payload, and
    nothing dangles as an unresolvable reference after decryption.
    """
    for i, obs in enumerate(observations):
        obs.setdefault("id", f"obs-{i + 1}")

    report: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",
        "code": {"text": result.suggested_title},
        "contained": observations,
        "result": [{"reference": f"#{obs['id']}"} for obs in observations],
        "extension": _provenance_extension(result.engine, result.mean_confidence),
    }
    if result.clinical_date:
        report["effectiveDateTime"] = result.clinical_date
    if result.facility:
        report["performer"] = [{"display": result.facility}]
    return report


def build(result: ExtractionResult) -> dict[str, Any]:
    """Pick the right resource for the document and attach it to `result.fhir`.

    One document produces exactly one top-level resource. The backend encrypts and
    stores that as a single record, and the patient verifies it as one thing.
    """
    date = result.clinical_date
    engine = result.engine

    labs_and_vitals = [e for e in result.entities if e.kind in ("lab", "vital")]
    meds = [e for e in result.entities if e.kind == "medication"]
    diagnoses = [e for e in result.entities if e.kind == "diagnosis"]

    if result.suggested_type == "lab" or (labs_and_vitals and not meds):
        observations = [build_observation(e, engine=engine, effective=date) for e in labs_and_vitals]
        result.fhir = (
            build_diagnostic_report(result, observations) if len(observations) != 1
            else observations[0]
        )

    elif result.suggested_type == "prescription" or meds:
        requests = [build_medication_request(e, engine=engine, authored=date) for e in meds]
        #: Several drugs on one prescription — a transaction-less collection Bundle keeps
        #: them together as the single document they physically are.
        result.fhir = requests[0] if len(requests) == 1 else {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": r} for r in requests],
        }

    elif diagnoses:
        conditions = [build_condition(e, engine=engine, onset=date) for e in diagnoses]
        result.fhir = conditions[0] if len(conditions) == 1 else {
            "resourceType": "Bundle", "type": "collection",
            "entry": [{"resource": c} for c in conditions],
        }

    else:
        #: Nothing structured survived, but the document is still the patient's record.
        #: A DocumentReference preserves it rather than throwing it away.
        result.fhir = {
            "resourceType": "DocumentReference",
            "status": "current",
            "docStatus": "preliminary",
            "description": result.suggested_title,
            "date": date,
            "extension": _provenance_extension(engine, result.mean_confidence),
        }

    return result.fhir
