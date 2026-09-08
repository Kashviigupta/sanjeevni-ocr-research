"""Module C — the physician-ready summary, fusing InterviewTranscript with
DocumentBundle. ayurvedic_assessment remains an honest gap pending AIIA review.
"""
from __future__ import annotations

from sanjeevani_ml.summary.assembler import build_history_summary
from sanjeevani_ml.summary.contradictions import detect_medication_contradictions
from sanjeevani_ml.summary.document_sections import (
    build_document_only_sections,
    build_drug_allergy_section,
    build_prior_investigations_section,
)
from sanjeevani_ml.summary.gaps import DocumentationGapsReport, build_documentation_gaps_report
from sanjeevani_ml.summary.interview_sections import build_interview_sections
from sanjeevani_ml.summary.translations import TEMPLATES, translate_template

__all__ = [
    "DocumentationGapsReport",
    "TEMPLATES",
    "build_document_only_sections",
    "build_documentation_gaps_report",
    "build_drug_allergy_section",
    "build_history_summary",
    "build_interview_sections",
    "build_prior_investigations_section",
    "detect_medication_contradictions",
    "translate_template",
]
