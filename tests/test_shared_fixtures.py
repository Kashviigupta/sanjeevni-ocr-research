"""The golden fixture must always parse against the real schemas.

`shared/fixtures/console-intake.json` is what Samridhi's summary screen renders in dev,
what Ronit's `/console/intakes/{id}` endpoint will serve first, and what Kashvi's
generator output is compared against. Three people rely on it being exactly the shape
the schemas say — so this test fails the moment either side drifts, and the fix is to
update fixture and schema in the same PR.
"""
from __future__ import annotations

import json
from pathlib import Path

from sanjeevani_ml.schemas import DocumentBundle, HistorySummary

FIXTURE = Path(__file__).resolve().parents[2] / "shared" / "fixtures" / "console-intake.json"


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestGoldenFixtureParses:
    def test_fixture_exists(self):
        assert FIXTURE.exists(), "shared/fixtures/console-intake.json is missing"

    def test_summary_parses_as_history_summary(self):
        summary = HistorySummary.model_validate(_load()["summary"])
        assert summary.status == "draft"          # machine output is never final
        assert summary.sections, "a summary with no sections renders an empty screen"

    def test_documents_parse_as_document_bundle(self):
        bundle = DocumentBundle.model_validate(_load()["documents"])
        assert bundle.timeline, "the fixture must exercise the timeline panel"

    def test_display_ref_is_not_phi(self):
        """The queue token is a desk number, never a name or an id a human carries."""
        ref = _load()["display_ref"]
        assert "sunita" not in ref.lower()
        assert not any(ch.isdigit() and len(ref) > 20 for ch in ref)


class TestFixtureExercisesEveryPanel:
    """The fixture exists to light up the whole summary screen. If a panel's data
    goes empty, Samridhi is silently testing less than she thinks."""

    def test_has_a_red_flag(self):
        assert _load()["summary"]["red_flags"], "RedFlagBanner needs data"

    def test_has_a_contradiction(self):
        assert _load()["summary"]["contradictions"], "ContradictionCards needs data"

    def test_has_abnormal_values(self):
        assert _load()["summary"]["abnormal_values"], "ClinicalAlerts needs data"

    def test_has_an_interaction(self):
        assert _load()["summary"]["interactions"], "ClinicalAlerts needs data"

    def test_has_exploration_including_not_explored(self):
        exploration = _load()["summary"]["exploration"]
        assert exploration["explored"] and exploration["not_explored"], (
            "ExplorationPanel must show what was covered AND what was not"
        )

    def test_has_gaps(self):
        assert _load()["summary"]["gaps"], "a summary that hides its gaps is misleading"

    def test_every_section_has_provenance(self):
        """Every claim traceable — the fixture must model the rule, not skip it."""
        for section in _load()["summary"]["sections"]:
            assert section["provenance"], f"section {section['heading']} has no provenance"

    def test_bilingual_content_everywhere(self):
        summary = _load()["summary"]
        for section in summary["sections"]:
            assert "en" in section["content"] and "hi" in section["content"]
        assert "hi" in summary["patient_confirmation"]

    def test_timeline_shows_a_deduplicated_document(self):
        entries = _load()["documents"]["timeline"]
        assert any(e["duplicate_of"] for e in entries), (
            "the dedup story is part of the demo — the fixture should carry one merged scan"
        )
