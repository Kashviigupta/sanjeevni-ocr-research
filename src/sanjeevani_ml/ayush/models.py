from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaptureMode(str, Enum):
    """How an AYUSH parameter's value is obtained.

    PATIENT_REPORTED: asked directly by the kiosk (voice or tap, per §3a/§3b).
    DERIVED: computed from other patient-reported answers; never asked directly.
    EXAMINER_REQUIRED: requires a physical exam; the kiosk never asks it,
        it is flagged for the physician instead.
    """

    PATIENT_REPORTED = "patient_reported"
    DERIVED = "derived"
    EXAMINER_REQUIRED = "examiner_required"


class AYUSHParameter(str, Enum):
    """The ten Dashavidha Pariksha parameters."""

    PRAKRITI = "prakriti"
    VIKRITI = "vikriti"
    SARA = "sara"
    SAMHANANA = "samhanana"
    PRAMANA = "pramana"
    SATMYA = "satmya"
    SATTVA = "sattva"
    AHARA_SHAKTI = "ahara_shakti"
    VYAYAMA_SHAKTI = "vyayama_shakti"
    VAYA = "vaya"


class AdditionalAYUSHParameter(str, Enum):
    """Additional AYUSH assessment parameters beyond the classic ten.

    NOTE: Samprapti is intentionally NOT a member here. Per the finalized
    §3b design, Samprapti is physician-synthesized during the consult
    (Module C's free-text field) and is not part of Module A's ontology
    at all — it is not asked, derived, or flagged for exam.
    """

    AGNI = "agni"
    KOSHTHA = "koshtha"
    AHARA_VIHARA = "ahara_vihara"
    NIDANA = "nidana"


# Single source of truth for who/what answers each parameter.
# Every member of AYUSHParameter and AdditionalAYUSHParameter must appear
# here exactly once. Samprapti is absent by design (see
# AdditionalAYUSHParameter's docstring above).
CAPTURE_MODE: dict[str, CaptureMode] = {
    # 9 patient-answerable, asked directly by the kiosk
    AYUSHParameter.VAYA.value: CaptureMode.PATIENT_REPORTED,
    AYUSHParameter.AHARA_SHAKTI.value: CaptureMode.PATIENT_REPORTED,
    AYUSHParameter.VYAYAMA_SHAKTI.value: CaptureMode.PATIENT_REPORTED,
    AYUSHParameter.SATMYA.value: CaptureMode.PATIENT_REPORTED,
    AYUSHParameter.SATTVA.value: CaptureMode.PATIENT_REPORTED,
    AdditionalAYUSHParameter.AGNI.value: CaptureMode.PATIENT_REPORTED,
    AdditionalAYUSHParameter.KOSHTHA.value: CaptureMode.PATIENT_REPORTED,
    AdditionalAYUSHParameter.AHARA_VIHARA.value: CaptureMode.PATIENT_REPORTED,
    AdditionalAYUSHParameter.NIDANA.value: CaptureMode.PATIENT_REPORTED,
    # 2 derived by formula from patient-reported proxy answers
    AYUSHParameter.PRAKRITI.value: CaptureMode.DERIVED,
    AYUSHParameter.VIKRITI.value: CaptureMode.DERIVED,
    # 3 examiner-required, kiosk never asks, flagged for physician
    AYUSHParameter.SARA.value: CaptureMode.EXAMINER_REQUIRED,
    AYUSHParameter.SAMHANANA.value: CaptureMode.EXAMINER_REQUIRED,
    AYUSHParameter.PRAMANA.value: CaptureMode.EXAMINER_REQUIRED,
}


@dataclass(frozen=True)
class AYUSHFieldStatus:
    parameter: str
    capture_mode: CaptureMode
    answered: bool
    clinically_validated: bool
