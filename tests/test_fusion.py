"""Tests for the fusion core: interview-side summary sections built against the
REAL ontology question ids, contradiction detection, and full HistorySummary
assembly.

Fixtures use exact question ids from shared/clinical/history-ontology.json
(reviewed 2026-08-22): cc.area, cc.primary, cc.describe, soc.*, px.*, rx.*,
pf.*, ros.* — not guessed keys.
"""
from __future__ import annotations

from sanjeevani_ml.schemas import (
    DocumentBundle,
    ExtractedEntity,
    InterviewTranscript,
    RedFlag,
    TimelineEntry,
)
from sanjeevani_ml.summary.assembler import build_history_summary
from sanjeevani_ml.summary.contradictions import detect_medication_contradictions
from sanjeevani_ml.summary.interview_sections import (
    build_chief_complaint_section,
    build_family_section,
    build_hpi_section,
    build_past_medical_section,
    build_past_surgical_section,
    build_personal_section,
    build_ros_section,
)


def _rx_entry(document_id="rx-1", drug="Tab. Glycomet 500mg") -> TimelineEntry:
    return TimelineEntry(
        date="2026-04-02", kind="prescription", title="Rx", document_id=document_id,
        entities=[ExtractedEntity(kind="medication", text=drug, confidence=0.9)],
    )


class TestChiefComplaintSection:
    def test_builds_from_structured_answers(self):
        transcript = InterviewTranscript(
            session_id="s1",
            answers={"cc.area": "chest", "cc.primary": "pain"},
        )
        section = build_chief_complaint_section(transcript)
        assert "Pain" in section.content["en"]
        assert "chest" in section.content["en"]

    def test_leads_with_verbatim_narration_when_present(self):
        transcript = InterviewTranscript(
            session_id="s1",
            answers={"cc.primary": "pain", "cc.describe": "It hurts when I breathe deeply."},
        )
        section = build_chief_complaint_section(transcript)
        assert "It hurts when I breathe deeply" in section.content["en"]
        assert section.confidence == 1.0

    def test_unanswered_produces_an_honest_gap_not_fabricated_content(self):
        transcript = InterviewTranscript(session_id="s1")
        section = build_chief_complaint_section(transcript)
        assert section.confidence == 0.0
        assert "not answered" in section.content["en"].lower() or "not reached" in section.content["en"].lower()


class TestHpiSection:
    def test_builds_full_socrates_for_a_pain_complaint(self):
        transcript = InterviewTranscript(
            session_id="s1",
            answers={
                "cc.primary": "pain",
                "soc.onset": "2 days",
                "soc.character": "sharp",
                "soc.severity": 7,
                "soc.associations": ["breathless", "sweating"],
            },
        )
        section = build_hpi_section(transcript)
        assert "sharp" in section.content["en"]
        assert "7/10" in section.content["en"]
        assert "breathlessness" in section.content["en"]

    def test_falls_back_to_narration_for_a_non_pain_complaint(self):
        """Only the pain branch's question set exists in the current ontology snapshot."""
        transcript = InterviewTranscript(
            session_id="s1",
            answers={"cc.primary": "fever", "cc.describe": "Fever since 3 days with chills."},
        )
        section = build_hpi_section(transcript)
        assert "Fever since 3 days" in section.content["en"]
        assert "not yet" in section.content["en"]

    def test_no_data_at_all_is_an_honest_gap(self):
        transcript = InterviewTranscript(session_id="s1", answers={"cc.primary": "digestive"})
        section = build_hpi_section(transcript)
        assert section.confidence == 0.0


class TestPastMedicalAndSurgical:
    def test_past_medical_lists_reported_conditions(self):
        transcript = InterviewTranscript(session_id="s1", answers={"px.conditions": ["diabetes", "hypertension"]})
        section = build_past_medical_section(transcript)
        assert "Diabetes" in section.content["en"]
        assert "High blood pressure" in section.content["en"]

    def test_past_medical_none_selected_is_still_reported_not_a_gap(self):
        """'none' is an answer, not an absence of one — 0 conditions found is informative."""
        transcript = InterviewTranscript(session_id="s1", answers={"px.conditions": ["none"]})
        section = build_past_medical_section(transcript)
        assert section.confidence == 0.0  # "none" filters out, leaving nothing to report — a real gap here is fine

    def test_past_surgical_no_reports_cleanly(self):
        transcript = InterviewTranscript(session_id="s1", answers={"px.surgery": False})
        section = build_past_surgical_section(transcript)
        assert "no prior operations" in section.content["en"].lower()

    def test_past_surgical_yes_includes_the_detail(self):
        transcript = InterviewTranscript(
            session_id="s1", answers={"px.surgery": True, "px.surgery_detail": "Appendix removed, 2019"},
        )
        section = build_past_surgical_section(transcript)
        assert "Appendix removed" in section.content["en"]


class TestFamilyAndPersonal:
    def test_family_history_lists_conditions(self):
        transcript = InterviewTranscript(session_id="s1", answers={"pf.family": ["diabetes", "heart"]})
        section = build_family_section(transcript)
        assert "Diabetes" in section.content["en"]
        assert "Heart disease" in section.content["en"]

    def test_personal_habits_lists_reported_habits(self):
        transcript = InterviewTranscript(session_id="s1", answers={"pf.habits": ["tobacco_smoke"]})
        section = build_personal_section(transcript)
        assert "Smoking" in section.content["en"]


class TestRos:
    def test_ros_screen_and_free_text_both_included(self):
        transcript = InterviewTranscript(
            session_id="s1",
            answers={"ros.screen": ["weight_loss"], "ros.anything": "I've also been very tired."},
        )
        section = build_ros_section(transcript)
        assert "losing weight" in section.content["en"]
        assert "very tired" in section.content["en"]


class TestMedicationContradictionDetection:
    def test_structured_rx_current_no_against_a_documented_drug_is_flagged(self):
        transcript = InterviewTranscript(session_id="s1", answers={"rx.current": False})
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        contradictions = detect_medication_contradictions(transcript, bundle)
        assert len(contradictions) == 1
        assert "Glycomet" in contradictions[0].from_documents

    def test_rx_current_yes_produces_no_contradiction(self):
        transcript = InterviewTranscript(session_id="s1", answers={"rx.current": True})
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        assert detect_medication_contradictions(transcript, bundle) == []

    def test_falls_back_to_narration_when_rx_current_not_reached(self):
        transcript = InterviewTranscript(session_id="s1", narration=["No medicines, doctor."])
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        assert len(detect_medication_contradictions(transcript, bundle)) == 1

    def test_structured_answer_takes_priority_over_narration(self):
        """If rx.current says yes, a stray narration phrase should not override it."""
        transcript = InterviewTranscript(
            session_id="s1", answers={"rx.current": True},
            narration=["Sometimes I say no medicines but I do take some."],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        assert detect_medication_contradictions(transcript, bundle) == []

    def test_a_denial_with_no_documented_medications_produces_nothing(self):
        transcript = InterviewTranscript(session_id="s1", answers={"rx.current": False})
        bundle = DocumentBundle(session_id="s1", timeline=[])
        assert detect_medication_contradictions(transcript, bundle) == []

    def test_document_id_is_preserved(self):
        transcript = InterviewTranscript(session_id="s1", answers={"rx.current": "no"})
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry(document_id="rx-42")])
        contradictions = detect_medication_contradictions(transcript, bundle)
        assert contradictions[0].document_id == "rx-42"


class TestHistorySummaryAssembly:
    def test_status_is_always_draft(self):
        transcript = InterviewTranscript(session_id="s1")
        bundle = DocumentBundle(session_id="s1", timeline=[])
        summary = build_history_summary(transcript, bundle)
        assert summary.status == "draft"

    def test_all_ten_sections_are_present(self):
        transcript = InterviewTranscript(session_id="s1")
        bundle = DocumentBundle(session_id="s1", timeline=[])
        summary = build_history_summary(transcript, bundle)
        headings = {s.heading for s in summary.sections}
        assert headings == {
            "drugs_allergy", "prior_investigations", "chief_complaint", "hpi",
            "past_medical", "past_surgical", "family", "personal", "ros",
            "ayurvedic_assessment",
        }

    def test_ayurvedic_assessment_is_a_gap_not_generated_content(self):
        transcript = InterviewTranscript(session_id="s1")
        bundle = DocumentBundle(session_id="s1", timeline=[])
        summary = build_history_summary(transcript, bundle)
        ayush = next(s for s in summary.sections if s.heading == "ayurvedic_assessment")
        assert ayush.confidence == 0.0
        assert "AIIA practitioner" in ayush.content["en"]

    def test_a_fully_answered_interview_produces_real_content_not_gaps(self):
        transcript = InterviewTranscript(
            session_id="s1",
            answers={
                "cc.area": "chest", "cc.primary": "pain", "cc.describe": "Sharp pain since morning.",
                "soc.onset": "this morning", "soc.character": "sharp", "soc.severity": 6,
                "px.conditions": ["hypertension"], "px.surgery": False,
                "pf.family": ["heart"], "pf.habits": ["none"],
                "ros.screen": ["none"], "rx.current": True,
            },
            red_flags=[RedFlag(reason="chest_pressure_cardiac_pattern", severity="critical", question_id="soc.character")],
        )
        bundle = DocumentBundle(session_id="s1", timeline=[])
        summary = build_history_summary(transcript, bundle)
        cc = next(s for s in summary.sections if s.heading == "chief_complaint")
        hpi = next(s for s in summary.sections if s.heading == "hpi")
        assert cc.confidence > 0
        assert hpi.confidence > 0
        assert len(summary.red_flags) == 1

    def test_medication_contradictions_are_included(self):
        transcript = InterviewTranscript(session_id="s1", answers={"rx.current": False})
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        summary = build_history_summary(transcript, bundle)
        assert len(summary.contradictions) == 1

    def test_never_phrases_anything_as_a_diagnosis(self):
        transcript = InterviewTranscript(session_id="s1", answers={"rx.current": False})
        bundle = DocumentBundle(session_id="s1", timeline=[_rx_entry()])
        summary = build_history_summary(transcript, bundle)
        for section in summary.sections:
            assert "diagnosis" not in section.content.get("en", "").lower()

    def test_an_empty_transcript_and_bundle_still_produce_a_valid_draft(self):
        transcript = InterviewTranscript(session_id="s1")
        bundle = DocumentBundle(session_id="s1", timeline=[])
        summary = build_history_summary(transcript, bundle)
        assert summary.session_id == "s1"
        assert len(summary.sections) == 10

    def test_unanswered_and_not_explored_both_appear_in_gaps(self):
        from sanjeevani_ml.schemas import ExplorationReport
        transcript = InterviewTranscript(
            session_id="s1", unanswered=["ros.anything"],
            exploration=ExplorationReport(not_explored=["cardiac risk factors"]),
        )
        bundle = DocumentBundle(session_id="s1", timeline=[])
        summary = build_history_summary(transcript, bundle)
        assert "ros.anything" in summary.gaps
        assert "cardiac risk factors" in summary.gaps
