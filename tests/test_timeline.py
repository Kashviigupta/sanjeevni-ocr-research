"""Timeline tests — Module B chronological organisation.

Fixtures are faithful transcriptions of the paperwork patients actually carry: a
pathology printout, the pharmacy's copy of the same prescription with two OCR errors in
it, a prescription from a different date, an undated handwritten slip, and a discharge
summary. Nothing here is a synthetic string built to match a regex.

Runs without Tesseract — text goes in directly, same as `test_pipeline.py`.
"""
from __future__ import annotations

from datetime import date

import pytest

from sanjeevani_ml.extraction.structurer import structure
from sanjeevani_ml.schemas import ExtractedEntity, OcrResult, TimelineEntry
from sanjeevani_ml.timeline.builder import build_timeline
from sanjeevani_ml.timeline.dating import find_date_candidates, resolve_clinical_date
from sanjeevani_ml.timeline.dedup import deduplicate, is_duplicate, jaccard
from sanjeevani_ml.timeline.ordering import order

TODAY = date(2026, 8, 21)

# --------------------------------------------------------------------- fixtures

LAB_MARCH = """
City Diagnostics Laboratory
123 MG Road, Pune                          Printed on: 16/03/2026
Patient: [redacted]        Collected on: 14/03/2026

Fasting Blood Sugar    142   mg/dL   (70-100)
HbA1c                   7.8  %       (4.0-5.6)
Total Cholesterol      210   mg/dL   (0-200)
"""

PRESCRIPTION_APRIL = """
Dr. A. Sharma, MBBS MD
Sunrise Clinic, Nashik
Date: 02/04/2026

Rx
Tab. Glycomet 500mg 1-0-1 x 30 days
Tab. Amlong 5mg 1-0-0 x 30 days
Cap. Omez 20mg 1-0-0 x 14 days

Review after 02/05/2026
"""

#: The pharmacy's carbon copy of the SAME prescription. Fainter, so the OCR mangles the
#: clinic name and one strength — this is what a second scan of one page really looks
#: like, and it is why dedup cannot be an equality check.
PRESCRIPTION_APRIL_COPY = """
Dr. A. Sharma, MBBS MD
Sunnse Clinic, Nashik
Date: 02/04/2026

Rx
Tab. Glycomet 500mg 1-0-1 x 30 days
Tab. Amlong 5mg 1-0-0 x 30 days
Cap. Omez 2Omg 1-0-0 x 14 days
"""

#: Same clinic, same drugs, one month later, with metformin stepped up. NOT a duplicate.
PRESCRIPTION_MAY_DOSE_CHANGE = """
Dr. A. Sharma, MBBS MD
Sunrise Clinic, Nashik
Date: 02/04/2026

Rx
Tab. Glycomet 1000mg 1-0-1 x 30 days
Tab. Amlong 5mg 1-0-0 x 30 days
Cap. Omez 20mg 1-0-0 x 14 days
"""

#: A torn handwritten slip with no date anywhere on it. Common, and it must not vanish.
UNDATED_SLIP = """
Tab. Ecosprin 75mg 0-0-1
Tab. Warf 5mg 0-0-1
"""

DISCHARGE_JANUARY = """
Government Medical College, Nagpur
Date of admission: 04/01/2026
Date of discharge: 09/01/2026

Diagnosis: Type 2 Diabetes Mellitus
"""


def _ocr(text: str, confidence: float = 0.92) -> OcrResult:
    return OcrResult(text=text, mean_confidence=confidence, engine="test-fixture")


def _entry(**kwargs) -> TimelineEntry:
    base = {"kind": "prescription", "title": "Prescription", "entities": []}
    return TimelineEntry(**{**base, **kwargs})


def _med(text: str, confidence: float = 0.9) -> ExtractedEntity:
    return ExtractedEntity(kind="medication", text=text, confidence=confidence)


# ------------------------------------------------------------------------ dating

class TestDating:
    def test_prefers_the_collection_date_over_the_print_date(self):
        """A lab is ABOUT the day the sample was drawn, not the day it was printed."""
        iso, confidence, _ = resolve_clinical_date(LAB_MARCH, today=TODAY)
        assert iso == "2026-03-14"
        assert confidence > 0.8

    def test_ignores_a_review_date(self):
        """`Review after 02/05/2026` is an instruction to the patient, not a document date."""
        iso, _, _ = resolve_clinical_date(PRESCRIPTION_APRIL, today=TODAY)
        assert iso == "2026-04-02"
        assert all(c.iso != "2026-05-02" for c in find_date_candidates(PRESCRIPTION_APRIL, today=TODAY))

    def test_reads_dd_mm_not_mm_dd(self):
        """03/04/2026 is 3 April in India. Reading it as 4 March is a clinical error."""
        iso, _, _ = resolve_clinical_date("Date: 03/04/2026", today=TODAY)
        assert iso == "2026-04-03"

    def test_never_swaps_day_and_month_to_rescue_a_date(self):
        """25/13/2026 is unusable. Silently reading it as 13 December invents a date."""
        iso, confidence, _ = resolve_clinical_date("Date: 25/13/2026", today=TODAY)
        assert iso is None
        assert confidence == 0.0

    def test_rejects_a_future_date(self):
        assert resolve_clinical_date("Date: 01/01/2030", today=TODAY)[0] is None

    def test_confidence_never_exceeds_ocr_confidence(self):
        """A date we misread is not made truer by a good label next to it."""
        _, confidence, _ = resolve_clinical_date(LAB_MARCH, ocr_confidence=0.4, today=TODAY)
        assert confidence <= 0.4

    def test_undated_document_returns_none_not_a_guess(self):
        iso, confidence, reason = resolve_clinical_date(UNDATED_SLIP, today=TODAY)
        assert iso is None
        assert confidence == 0.0
        assert "no usable date" in reason


# ------------------------------------------------------------------------ dedup

class TestDeduplication:
    def test_merges_a_carbon_copy_of_the_same_prescription(self):
        a = _entry(date="2026-04-02", document_id="doc-1",
                   entities=[_med("Tab. Glycomet 500mg"), _med("Tab. Amlong 5mg"), _med("Cap. Omez 20mg")])
        b = _entry(date="2026-04-02", document_id="doc-2",
                   entities=[_med("Tab. Glycomet 500mg"), _med("Tab. Amlong 5mg"), _med("Cap. Omez 2Omg")])
        duplicate, _ = is_duplicate(a, b)
        assert duplicate

    def test_a_merge_never_loses_the_second_scan(self):
        """Shortening the timeline must not make a document unopenable."""
        a = _entry(date="2026-04-02", document_id="doc-1",
                   entities=[_med("Tab. Glycomet 500mg", 0.95), _med("Tab. Amlong 5mg", 0.95)])
        b = _entry(date="2026-04-02", document_id="doc-2",
                   entities=[_med("Tab. Glycomet 500mg", 0.6), _med("Tab. Amlong 5mg", 0.6)])
        kept, notes = deduplicate([a, b])
        assert len(kept) == 1
        assert kept[0].document_id == "doc-1"        # the better scan leads
        assert "doc-2" in kept[0].duplicate_of       # the other is still reachable
        assert len(notes) == 1

    def test_a_dose_change_is_never_merged_away(self):
        """Glycomet 500 -> 1000 may be a real dose change. Surface both (Kashvi's call)."""
        a = _entry(date="2026-04-02", document_id="doc-1",
                   entities=[_med("Tab. Glycomet 500mg"), _med("Tab. Amlong 5mg"), _med("Cap. Omez 20mg")])
        b = _entry(date="2026-04-02", document_id="doc-2",
                   entities=[_med("Tab. Glycomet 1000mg"), _med("Tab. Amlong 5mg"), _med("Cap. Omez 20mg")])
        duplicate, reason = is_duplicate(a, b)
        assert not duplicate
        assert "dose change" in reason

    def test_different_dates_are_never_merged(self):
        """Collection vs report date is indistinguishable from two draws. Keep separate."""
        a = _entry(kind="lab", date="2026-03-14", document_id="doc-1",
                   entities=[_med("Fasting Blood Sugar"), _med("HbA1c")])
        b = _entry(kind="lab", date="2026-03-15", document_id="doc-2",
                   entities=[_med("Fasting Blood Sugar"), _med("HbA1c")])
        assert not is_duplicate(a, b)[0]

    def test_undated_documents_are_never_merged_into_each_other(self):
        a = _entry(document_id="doc-1", entities=[_med("Tab. Ecosprin 75mg"), _med("Tab. Warf 5mg")])
        b = _entry(document_id="doc-2", entities=[_med("Tab. Ecosprin 75mg"), _med("Tab. Warf 5mg")])
        assert not is_duplicate(a, b)[0]

    def test_a_lab_and_a_prescription_are_never_merged(self):
        a = _entry(kind="lab", date="2026-04-02", entities=[_med("HbA1c"), _med("FBS")])
        b = _entry(kind="prescription", date="2026-04-02", entities=[_med("HbA1c"), _med("FBS")])
        assert not is_duplicate(a, b)[0]

    def test_one_shared_entity_is_not_evidence_of_duplication(self):
        """Two documents mentioning one common drug are not the same document."""
        a = _entry(date="2026-04-02", entities=[_med("Tab. Glycomet 500mg")])
        b = _entry(date="2026-04-02", entities=[_med("Tab. Glycomet 500mg")])
        assert not is_duplicate(a, b)[0]

    def test_jaccard_is_zero_on_an_empty_side(self):
        assert jaccard(set(), {"a", "b"}) == 0.0


# --------------------------------------------------------------------- ordering

class TestOrdering:
    def test_oldest_first(self):
        entries = [
            _entry(date="2026-04-02", title="Prescription"),
            _entry(kind="note", date="2026-01-09", title="Discharge"),
            _entry(kind="lab", date="2026-03-14", title="Lab"),
        ]
        assert [e.date for e in order(entries)] == ["2026-01-09", "2026-03-14", "2026-04-02"]

    def test_undated_documents_trail_and_are_never_interleaved(self):
        entries = [
            _entry(document_id="undated", title="Handwritten slip"),
            _entry(kind="lab", date="2026-03-14", title="Lab"),
        ]
        ordered = order(entries)
        assert ordered[-1].document_id == "undated"

    def test_investigations_precede_the_prescription_written_the_same_day(self):
        entries = [
            _entry(date="2026-04-02", title="Prescription"),
            _entry(kind="lab", date="2026-04-02", title="Lab"),
        ]
        assert order(entries)[0].kind == "lab"


# ------------------------------------------------------------ end-to-end bundle

def _bundle(texts: list[str], ids: list[str], confidence: float = 0.92):
    results = [structure(_ocr(t, confidence)) for t in texts]
    return build_timeline(
        results, session_id="sess-1", document_ids=ids, source_texts=texts, today=TODAY
    )


class TestBundle:
    def test_a_shoebox_of_documents_comes_back_in_date_order(self):
        bundle = _bundle(
            [PRESCRIPTION_APRIL, DISCHARGE_JANUARY, LAB_MARCH],
            ["rx-apr", "dis-jan", "lab-mar"],
        )
        dated = [e.date for e in bundle.timeline if e.date]
        assert dated == sorted(dated)
        assert dated[0] == "2026-01-04"

    def test_the_undated_slip_survives(self):
        """A record the patient handed us is never silently dropped."""
        bundle = _bundle([PRESCRIPTION_APRIL, UNDATED_SLIP], ["rx-apr", "slip"])
        assert any(e.document_id == "slip" for e in bundle.timeline)
        assert bundle.timeline[-1].date is None

    def test_interactions_are_checked_across_documents_not_within_one(self):
        """The dangerous pair usually comes from two different prescribers."""
        bundle = _bundle([PRESCRIPTION_APRIL, UNDATED_SLIP], ["rx-apr", "slip"])
        assert any(a.severity == "major" for a in bundle.interactions)

    def test_an_unreadable_page_is_offered_for_retake_not_dropped(self):
        results = [structure(_ocr(LAB_MARCH)), structure(_ocr("", 0.0))]
        bundle = build_timeline(
            results, session_id="sess-1", document_ids=["lab", "blur"], today=TODAY
        )
        assert bundle.unreadable == ["blur"]
        assert bundle.documents_processed == 2

    def test_abnormal_values_are_now_populated_from_lab_entities(self):
        """Superseded expectation: abnormal-value highlighting is implemented now.
        LAB_MARCH's fasting blood sugar (142, range 70-100) is genuinely abnormal.
        """
        bundle = _bundle([LAB_MARCH], ["lab-mar"])
        assert any(a.analyte == "Fasting Blood Sugar" and a.direction == "high"
                   for a in bundle.abnormal_values)

    def test_mismatched_ids_raise_rather_than_mislabel_a_document(self):
        """Pairing a scan with the wrong id attaches it to the wrong history."""
        with pytest.raises(ValueError):
            build_timeline([structure(_ocr(LAB_MARCH))], session_id="s", document_ids=[])

    def test_no_documents_returns_an_empty_bundle_not_a_crash(self):
        bundle = build_timeline([], session_id="sess-1", document_ids=[], today=TODAY)
        assert bundle.timeline == []
        assert bundle.mean_confidence == 0.0
