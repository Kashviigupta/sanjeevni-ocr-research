from __future__ import annotations

from sanjeevani_ml.ayush.models import (
    AYUSHParameter,
    AdditionalAYUSHParameter,
    CaptureMode,
    CAPTURE_MODE,
)


DASHAVIDHA_ORDER: tuple[str, ...] = (
    AYUSHParameter.PRAKRITI.value,
    AYUSHParameter.VIKRITI.value,
    AYUSHParameter.SARA.value,
    AYUSHParameter.SAMHANANA.value,
    AYUSHParameter.PRAMANA.value,
    AYUSHParameter.SATMYA.value,
    AYUSHParameter.SATTVA.value,
    AYUSHParameter.AHARA_SHAKTI.value,
    AYUSHParameter.VYAYAMA_SHAKTI.value,
    AYUSHParameter.VAYA.value,
)


# Samprapti removed: it is excluded from Module A's ontology entirely
# (physician-synthesized free-text field in Module C). See models.py.
ADDITIONAL_AYUSH_ORDER: tuple[str, ...] = (
    AdditionalAYUSHParameter.AGNI.value,
    AdditionalAYUSHParameter.KOSHTHA.value,
    AdditionalAYUSHParameter.AHARA_VIHARA.value,
    AdditionalAYUSHParameter.NIDANA.value,
)


def dashavidha_parameters() -> tuple[str, ...]:
    """Return the required Dashavidha parameters in configured order."""
    return DASHAVIDHA_ORDER


def additional_parameters() -> tuple[str, ...]:
    """Return additional AYUSH assessment parameters."""
    return ADDITIONAL_AYUSH_ORDER


def known_parameter_names() -> tuple[str, ...]:
    """Return every valid AYUSH parameter name (Dashavidha + additional).

    Samprapti is intentionally absent — it is not a Module A parameter.
    """
    return DASHAVIDHA_ORDER + ADDITIONAL_AYUSH_ORDER


def capture_mode_of(parameter: str) -> CaptureMode:
    """Return how `parameter` is captured.

    Raises KeyError for any name outside known_parameter_names(), which
    includes "samprapti" — that lookup is expected to fail loudly rather
    than silently return a default, since Samprapti should never reach
    this layer at all.
    """
    return CAPTURE_MODE[parameter]


def patient_reported_parameters() -> tuple[str, ...]:
    """Parameters the kiosk asks directly, in ontology order."""
    return tuple(
        parameter
        for parameter in known_parameter_names()
        if capture_mode_of(parameter) == CaptureMode.PATIENT_REPORTED
    )


def derived_parameters() -> tuple[str, ...]:
    """Parameters computed from other answers; never asked directly."""
    return tuple(
        parameter
        for parameter in known_parameter_names()
        if capture_mode_of(parameter) == CaptureMode.DERIVED
    )


def examiner_required_parameters() -> tuple[str, ...]:
    """Parameters that require a physical exam; flagged for the physician."""
    return tuple(
        parameter
        for parameter in known_parameter_names()
        if capture_mode_of(parameter) == CaptureMode.EXAMINER_REQUIRED
    )