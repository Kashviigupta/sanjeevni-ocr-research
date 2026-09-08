"""Tests that `hint` is a real signal, not a silently-ignored field.

Reviewer caught: `hint` existed on ExtractRequest/DocumentIngestRequest but
`structure()` dropped the parameter entirely, so a patient tapping "this is a
lab report" had zero effect. This file proves hint actually changes the
outcome, at both the unit level (structure()) and the HTTP level (the three
endpoints that accept a hint).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from sanjeevani_ml.extraction.structurer import structure
from sanjeevani_ml.schemas import OcrResult
from sanjeevani_ml.service.main import app

client = TestClient(app)

#: Deliberately ambiguous text: no drug lines, no lab rows, no diagnosis keywords
#: — nothing for the entity-shape heuristic (`_classify`) to grab onto, so its
#: only honest answer is "note". A hint is the only way this ever becomes
#: anything more specific.
AMBIGUOUS_TEXT = "City Clinic\nDate: 02/04/2026\nPlease follow up in two weeks."

LAB_TEXT_WITH_HINT_MISMATCH = "Some handwritten note the OCR barely read."


def _ocr(text: str) -> OcrResult:
    return OcrResult(text=text, mean_confidence=0.9, engine="test-fixture")


class TestStructureHint:
    def test_no_hint_falls_back_to_the_entity_shape_heuristic(self):
        result = structure(_ocr(AMBIGUOUS_TEXT))
        assert result.suggested_type == "note"  # nothing for _classify to grab onto

    def test_a_hint_overrides_the_heuristic_on_ambiguous_text(self):
        """The whole point: a patient's own statement is a real, cheap signal."""
        result = structure(_ocr(AMBIGUOUS_TEXT), hint="lab")
        assert result.suggested_type == "lab"

    def test_a_hint_overrides_even_when_entities_would_suggest_something_else(self):
        """Patient's own word wins outright — see structure()'s docstring for why."""
        rx_text = "Tab. Glycomet 500mg 1-0-1 x 30 days"  # entities would say "prescription"
        result = structure(_ocr(rx_text), hint="lab")
        assert result.suggested_type == "lab"

    def test_no_hint_still_classifies_correctly_from_entities(self):
        """Confirms the fix didn't break the no-hint path."""
        rx_text = "Tab. Glycomet 500mg 1-0-1 x 30 days"
        result = structure(_ocr(rx_text))
        assert result.suggested_type == "prescription"


class TestHintOverHttp:
    def test_extract_endpoint_honours_the_hint(self):
        response = client.post("/extract", json={"text": AMBIGUOUS_TEXT, "hint": "lab"})
        assert response.status_code == 200
        assert response.json()["suggested_type"] == "lab"

    def test_documents_ingest_endpoint_honours_the_hint(self):
        response = client.post("/documents/ingest", json={
            "document_id": "doc-1", "text": AMBIGUOUS_TEXT, "hint": "condition",
        })
        assert response.status_code == 200
        assert response.json()["kind"] == "condition"

    def test_documents_timeline_endpoint_honours_a_per_document_hint(self):
        """Each document in the batch carries its own hint — not one shared
        for the whole request, since different documents in one upload batch
        can be different types."""
        response = client.post("/documents/timeline", json={
            "session_id": "sess-1",
            "documents": [
                {"document_id": "doc-1", "text": AMBIGUOUS_TEXT, "hint": "lab"},
                {"document_id": "doc-2", "text": AMBIGUOUS_TEXT, "hint": "prescription"},
            ],
        })
        assert response.status_code == 200
        entries = {e["document_id"]: e["kind"] for e in response.json()["timeline"]}
        assert entries["doc-1"] == "lab"
        assert entries["doc-2"] == "prescription"

    def test_no_hint_over_http_still_works_as_before(self):
        response = client.post("/extract", json={"text": "Tab. Glycomet 500mg 1-0-1 x 30 days"})
        assert response.json()["suggested_type"] == "prescription"
