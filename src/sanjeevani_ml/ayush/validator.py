from __future__ import annotations

from sanjeevani_ml.ayush.dashavidha import (
    DASHAVIDHA_ORDER,
    ADDITIONAL_AYUSH_ORDER,
    capture_mode_of,
)
from sanjeevani_ml.ayush.models import CaptureMode


class AYUSHValidator:
    """Structural validation for AYUSH interview data.

    This validator does NOT clinically interpret answers.
    It only checks whether required ontology fields were captured.

    Per §3b's capture_mode split:
    - patient_reported parameters count toward "missing" if unanswered.
    - derived parameters (Prakriti, Vikriti) are never "missing" — they're
      computed, not answered, and are skipped here entirely.
    - examiner_required parameters (Sara, Samhanana, Pramana) are never
      "missing" either — they're always pending_physician_review(),
      never counted against interview completeness.
    """

    def __init__(self, answers: dict[str, object]):
        self.answers = answers

    def answered(self, question_id: str) -> bool:
        return question_id in self.answers

    def _missing_patient_reported(
        self,
        parameter_order: tuple[str, ...],
        parameter_question_map: dict[str, str],
    ) -> list[str]:
        missing = []

        for parameter in parameter_order:
            if capture_mode_of(parameter) != CaptureMode.PATIENT_REPORTED:
                continue

            question_id = parameter_question_map.get(parameter)

            if question_id is None:
                missing.append(parameter)
                continue

            if not self.answered(question_id):
                missing.append(parameter)

        return missing

    def missing_dashavidha(
        self,
        parameter_question_map: dict[str, str],
    ) -> list[str]:
        return self._missing_patient_reported(
            DASHAVIDHA_ORDER, parameter_question_map
        )

    def missing_additional(
        self,
        parameter_question_map: dict[str, str],
    ) -> list[str]:
        return self._missing_patient_reported(
            ADDITIONAL_AYUSH_ORDER, parameter_question_map
        )

    def pending_physician_review(self) -> tuple[str, ...]:
        """Examiner-required parameters, always returned as pending.

        These are never counted as "missing" — they were never askable
        in the first place. Module C should render these as
        "pending exam," not blank.
        """
        return tuple(
            parameter
            for parameter in DASHAVIDHA_ORDER
            if capture_mode_of(parameter) == CaptureMode.EXAMINER_REQUIRED
        )

    def is_complete(
        self,
        parameter_question_map: dict[str, str],
    ) -> bool:
        """True once every patient_reported parameter has been answered.

        Derived and examiner_required parameters are intentionally
        excluded from this check — see class docstring.
        """
        return not (
            self.missing_dashavidha(parameter_question_map)
            or self.missing_additional(parameter_question_map)
        )
