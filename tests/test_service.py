"""Tests for the three endpoints that make everything built so far reachable over
HTTP: /documents/ingest, /documents/timeline, /summary/generate.

Every request uses `text` (never `data_base64`), so these tests never touch
Tesseract or PyMuPDF — they exercise the real extraction/terminology/timeline/
summary pipeline through the actual HTTP layer, exactly as the backend will call
it, without needing a working OCR install in CI.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from sanjeevani_ml.service.main import app

client = TestClient(app)

LAB_TEXT = """
City Diagnostics Laboratory
Collected on: 14/03/2026
Fasting Blood Sugar    142   mg/dL   (70-100)
"""

PRESCRIPTION_TEXT = """
Dr. A. Sharma, MBBS MD
Date: 02/04/2026
Tab. Glycomet 500mg 1-0-1 x 30 days
"""


class TestDocumentsIngest:
    def test_ingests_a_single_document_into_a_timeline_entry(self):
        response = client.post("/documents/ingest", json={
            "document_id": "lab-1", "text": LAB_TEXT,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == "lab-1"
        assert body["kind"] == "lab"

    def test_missing_text_and_base64_is_a_400_not_a_500(self):
        response = client.post("/documents/ingest", json={"document_id": "empty-1"})
        assert response.status_code == 400

    def test_the_date_is_extracted_correctly(self):
        response = client.post("/documents/ingest", json={
            "document_id": "lab-1", "text": LAB_TEXT,
        })
        assert response.json()["date"] == "2026-03-14"


class TestDocumentsTimeline:
    def test_builds_a_bundle_from_multiple_documents(self):
        response = client.post("/documents/timeline", json={
            "session_id": "sess-1",
            "documents": [
                {"document_id": "lab-1", "text": LAB_TEXT},
                {"document_id": "rx-1", "text": PRESCRIPTION_TEXT},
            ],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["documents_processed"] == 2
        assert len(body["timeline"]) == 2

    def test_the_bundle_is_dated_oldest_first(self):
        response = client.post("/documents/timeline", json={
            "session_id": "sess-1",
            "documents": [
                {"document_id": "rx-1", "text": PRESCRIPTION_TEXT},  # April
                {"document_id": "lab-1", "text": LAB_TEXT},           # March
            ],
        })
        dates = [e["date"] for e in response.json()["timeline"] if e["date"]]
        assert dates == sorted(dates)

    def test_abnormal_values_are_flagged_through_the_http_layer(self):
        response = client.post("/documents/timeline", json={
            "session_id": "sess-1",
            "documents": [{"document_id": "lab-1", "text": LAB_TEXT}],
        })
        abnormal = response.json()["abnormal_values"]
        assert any(a["analyte"] == "Fasting Blood Sugar" for a in abnormal)

    def test_an_empty_documents_list_is_rejected_not_silently_accepted(self):
        response = client.post("/documents/timeline", json={
            "session_id": "sess-1", "documents": [],
        })
        assert response.status_code == 422  # min_length=1 on the schema


class TestSummaryGenerate:
    def test_generates_a_draft_summary_from_transcript_and_bundle(self):
        timeline_response = client.post("/documents/timeline", json={
            "session_id": "sess-1",
            "documents": [{"document_id": "rx-1", "text": PRESCRIPTION_TEXT}],
        })
        bundle = timeline_response.json()

        response = client.post("/summary/generate", json={
            "transcript": {"session_id": "sess-1", "answers": {"rx.current": True}},
            "bundle": bundle,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "draft"
        assert body["session_id"] == "sess-1"

    def test_all_ten_sections_reach_the_client_over_http(self):
        response = client.post("/summary/generate", json={
            "transcript": {"session_id": "sess-1"},
            "bundle": {"session_id": "sess-1", "timeline": []},
        })
        headings = {s["heading"] for s in response.json()["sections"]}
        assert len(headings) == 10

    def test_a_medication_contradiction_survives_the_full_http_round_trip(self):
        timeline_response = client.post("/documents/timeline", json={
            "session_id": "sess-1",
            "documents": [{"document_id": "rx-1", "text": PRESCRIPTION_TEXT}],
        })
        bundle = timeline_response.json()

        response = client.post("/summary/generate", json={
            "transcript": {"session_id": "sess-1", "answers": {"rx.current": False}},
            "bundle": bundle,
        })
        assert len(response.json()["contradictions"]) >= 1

    def test_never_returns_anything_but_draft_status(self):
        response = client.post("/summary/generate", json={
            "transcript": {"session_id": "sess-1"},
            "bundle": {"session_id": "sess-1", "timeline": []},
        })
        assert response.json()["status"] == "draft"


class TestEndToEndPipeline:
    def test_ingest_then_timeline_then_summary_end_to_end_over_http(self):
        """The full path the backend will actually walk, start to finish."""
        timeline_response = client.post("/documents/timeline", json={
            "session_id": "sess-e2e",
            "documents": [
                {"document_id": "lab-1", "text": LAB_TEXT},
                {"document_id": "rx-1", "text": PRESCRIPTION_TEXT},
            ],
        })
        assert timeline_response.status_code == 200

        summary_response = client.post("/summary/generate", json={
            "transcript": {
                "session_id": "sess-e2e",
                "answers": {"cc.primary": "digestive", "rx.current": True},
            },
            "bundle": timeline_response.json(),
        })
        assert summary_response.status_code == 200
        summary = summary_response.json()
        assert summary["status"] == "draft"
        assert any(a["direction"] == "high" for a in summary["abnormal_values"])
