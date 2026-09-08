from __future__ import annotations

from sanjeevani_ml.ayush.dashavidha import (
    DASHAVIDHA_ORDER,
    ADDITIONAL_AYUSH_ORDER,
    capture_mode_of,
)
from sanjeevani_ml.ayush.models import CaptureMode


class AYUSHFlow:
    """Controls ordering of AYUSH-specific ontology sections.

    Clinical content remains in the ontology and must be
    practitioner-reviewed.

    Per §3b: AYUSH is a separate, fixed-order block walked as its own
    guaranteed section, using the same voice/tap interaction mechanism
    as the rest of the interview (§3a) — no AYUSH-specific interaction
    state. This class is also the enforcement point for capture_mode:
    the kiosk must only ever generate questions for patient_reported
    parameters, never for derived or examiner_required ones.
    """

    def __init__(self, ontology: dict):
        self.ontology = ontology

    def is_ayush_section(self, section: dict) -> bool:
        return section.get("mode") == "ayush"

    def section_order(self, section: dict) -> int:
        return int(section.get("order", 0))

    def is_askable(self, section: dict) -> bool:
        """True if this section's parameter should generate a kiosk question.

        Sections whose parameter is derived or examiner_required must
        never surface as a question — this is what actually stops the
        kiosk from ever quietly asking about Sara.
        """
        parameter = section.get("parameter")

        if parameter is None:
            return False

        try:
            return capture_mode_of(parameter) == CaptureMode.PATIENT_REPORTED
        except KeyError:
            # Unknown parameter (e.g. samprapti, or a typo) — never askable.
            return False

    def ordered_sections(self, sections: list[dict]) -> list[dict]:
        """Return AYUSH sections that should be asked, in fixed order.

        Filters to patient_reported parameters only, per capture_mode.
        Derived and examiner_required sections are excluded from the
        kiosk's question flow entirely — they're surfaced elsewhere
        (derived values are computed; examiner_required ones become
        pending_physician_review() items via AYUSHValidator).
        """
        askable_sections = [
            section
            for section in sections
            if self.is_ayush_section(section) and self.is_askable(section)
        ]

        return sorted(
            askable_sections,
            key=self.section_order,
        )

    @staticmethod
    def should_preserve_order(section: dict) -> bool:
        """AYUSH sections should be explicitly marked as fixed-order."""

        return (
            section.get("mode") == "ayush"
            and section.get("ordering") == "fixed"
        )
