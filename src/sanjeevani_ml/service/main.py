"""The HTTP surface — shared by Mahek and Kashvi.

**The backend is the only caller.** No frontend talks to this service directly: the
backend owns authentication, consent and the audit ledger, so every AI result must reach
a human through it.

This service is deliberately stateless and has no database. Bytes in, structure out.
Nothing is persisted, which means nothing here can leak a patient record.
"""
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

import base64
import binascii
import logging
import threading

from fastapi import FastAPI, HTTPException, status

from ..config import settings
from pathlib import Path


from ..dialogue.engine import DialogueEngine
from ..extraction.structurer import structure
from ..fhir.builders import build
from ..interactions.checker import check, summarize
from ..ocr import engine as ocr_engine
from ..speech import asr, tts
from ..speech.bhashini_client import BhashiniError
from ..schemas import (
    AsrRequest,
    AsrResponse,
    DocumentBundle,
    DocumentIngestRequest,
    ExtractRequest,
    ExtractionResult,
    HistorySummary,
    InteractionRequest,
    InterviewNextIn,
    InterviewNextOut,
    OcrRequest,
    OcrResult,
    ServiceStatus,
    SummaryGenerateRequest,
    TimelineEntry,
    TimelineRequest,
    TtsRequest,
    TtsResponse,
)
from ..summary.assembler import build_history_summary
from ..terminology.loader import reload_tables, term_count
from ..terminology.mapper import code_result
from ..timeline.builder import build_timeline
from ..timeline.dating import resolve_clinical_date

#: shared/ sits beside ml/ in the repo and beside the app in the image.
#: In the repo, shared/ is four levels up from service/main.py; in the Docker image
#: it sits beside the package. Resolve the first that exists rather than assuming.
def _find_shared(filename: str) -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "shared" / "clinical" / filename
        if candidate.exists():
            return candidate
    # Fail here, where the message can name what was searched, rather than deferring
    # to first use — a kiosk failing mid-interview with "file not found" tells whoever
    # is on shift nothing they can act on.
    raise FileNotFoundError(
        f"clinical data file not found. Expected shared/clinical/{filename} "
        f"in one of: {[str(p) for p in list(here.parents)[:4]]}"
    )


ONTOLOGY_PATH = _find_shared("history-ontology.json")

#: Which questions become relevant once an answer makes a condition plausible. This is
#: what makes the interview feel like a clinician rather than a form: say "chest pain"
#: and the next questions are about radiation and exertion, not your family history.
#:
#: It ORDERS questions and nothing else. It is not a diagnosis and must never be
#: presented as one.
CONDITION_MAP_PATH = _find_shared("condition-map.json")

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("sanjeevani_ml")

app = FastAPI(
    title="Sanjeevani ML service",
    version="1.0.0",
    description=(
        "OCR, clinical extraction and terminology coding.\n\n"
        "**Everything returned here is a DRAFT.** The patient verifies it in the app "
        "before it becomes part of their record. Confidence scores are reported honestly "
        "and are meant to be shown to the user, not hidden.\n\n"
        "Called by the backend only - never exposed to a frontend."
    ),
)


def _decode(data_base64: str) -> bytes:
    try:
        raw = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "data_base64 is not valid base64")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "document exceeds 10 MB")
    return raw


def _resolve_source(doc: DocumentIngestRequest | ExtractRequest) -> OcrResult:
    """The same text-or-base64 resolution `extract()` uses, factored out so
    `/documents/ingest` and `/documents/timeline` do not duplicate it.
    """
    if doc.text:
        return OcrResult(text=doc.text, mean_confidence=0.99, engine="text-input")
    if doc.data_base64:
        return ocr_engine.ocr_document(
            _decode(doc.data_base64), media_type=doc.media_type, language=doc.language
        )
    document_label = getattr(doc, "document_id", None)
    prefix = f"document {document_label}: " if document_label else ""
    raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{prefix}provide either text or data_base64")


@app.get("/health", response_model=ServiceStatus, tags=["ops"])
def health() -> ServiceStatus:
    """Honest capability report. `degraded` is a normal state, not an outage.

    Without Tesseract the service still runs and the app falls back to manual entry, so
    the rest of the team is never blocked on one person's local setup.
    """
    version = ocr_engine.tesseract_version()
    return ServiceStatus(
        status="ok" if version else "degraded",
        tesseract=bool(version),
        tesseract_version=version,
        vision_llm=settings.vision_enabled,
        terminology_terms=term_count(),
        detail=None if version else "tesseract binary not found - OCR endpoints return empty text",
    )


@app.post("/ocr", response_model=OcrResult, tags=["ocr"])
def ocr(body: OcrRequest) -> OcrResult:
    """Document to text with per-word confidence. **Mahek's endpoint.**

    Per-word confidence is the point: it lets the app highlight exactly which fields the
    patient should double-check, instead of asking them to re-read everything.
    """
    return ocr_engine.ocr_document(
        _decode(body.data_base64), media_type=body.media_type, language=body.language
    )


@app.post("/speech/transcribe", response_model=AsrResponse, tags=["speech"])
def transcribe(body: AsrRequest) -> AsrResponse:
    """A spoken answer to text. **Mahek's endpoint.**

    This is the accessibility argument made real: a patient who cannot read finishes
    the interview by speaking, in their own language, in a noisy OPD corridor.

    503 rather than 500 when Bhashini is unreachable or unconfigured, because the kiosk
    is expected to fall back to touch and carry on. Losing voice makes the interview
    harder; it must never end it.
    """
    try:
        result = asr.transcribe(
            _decode(body.data_base64),
            language=body.language,
            audio_format=body.audio_format,
        )
    except BhashiniError as e:
        log.warning("asr unavailable: %s", e)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "speech recognition is unavailable - the patient can still answer by touch",
        ) from e

    # Never log the transcript. It is the patient's own words about their symptoms.
    log.info("asr: %s chars, confidence=%s", len(result.text), result.confidence)
    return AsrResponse(text=result.text, confidence=result.confidence,
                       language=body.language)


@app.post("/speech/speak", response_model=TtsResponse, tags=["speech"])
def speak(body: TtsRequest) -> TtsResponse:
    """Prompt text to spoken audio. **Mahek's endpoint.**

    Every question is spoken aloud, so a non-reading patient can complete the whole
    interview by ear. The client asks for the prompt it was just handed rather than
    the server pushing audio, because the client knows when the patient is ready to
    hear it.
    """
    try:
        result = tts.synthesize(body.text, language=body.language, gender=body.gender)
    except ValueError as e:
        # An unsupported language is the caller's mistake, not an outage.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    except BhashiniError as e:
        log.warning("tts unavailable: %s", e)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "speech synthesis is unavailable - the prompt can still be read on screen",
        ) from e

    return TtsResponse(
        audio_base64=base64.b64encode(result.audio_bytes).decode(),
        language=result.language,
    )


@app.post("/extract", response_model=ExtractionResult, tags=["extraction"])
def extract(body: ExtractRequest) -> ExtractionResult:
    """The full pipeline: OCR (if needed) to entities to codes to FHIR.

    **This is what the backend actually calls** on `POST /records/upload`. The result
    becomes a `draft` record the patient reviews. Nothing here is clinical truth yet.
    """
    source = _resolve_source(body)
    result = structure(source, hint=body.hint)
    result = code_result(result)
    build(result)

    log.info(  # counts and scores only - never the extracted text (PHI)
        "extract: type=%s entities=%d confidence=%.2f engine=%s",
        result.suggested_type, len(result.entities), result.mean_confidence, result.engine,
    )
    return result


@app.post("/documents/ingest", response_model=TimelineEntry, tags=["documents"])
def documents_ingest(body: DocumentIngestRequest) -> TimelineEntry:
    """One uploaded document -> one dated, coded, FHIR-built TimelineEntry.

    The single-document building block. For a whole session's documents assembled
    into one ordered, de-duplicated bundle, use `/documents/timeline` instead —
    it runs this same pipeline per document internally.
    """
    source = _resolve_source(body)
    result = structure(source, hint=body.hint)
    result = code_result(result)
    build(result)

    if result.clinical_date:
        # structure() already found and scored a date — trust its own confidence
        # rather than deriving a second, possibly disagreeing number for the
        # same fact.
        entry_date, date_confidence = result.clinical_date, result.mean_confidence
    else:
        entry_date, date_confidence, _reason = resolve_clinical_date(
            source.text, ocr_confidence=result.mean_confidence
        )

    entry = TimelineEntry(
        date=entry_date,
        date_confidence=date_confidence,
        kind=result.suggested_type,
        title=result.suggested_title,
        facility=result.facility,
        document_id=body.document_id,
        entities=result.entities,
        fhir=result.fhir,
    )
    log.info(  # counts and scores only — never extracted text (PHI)
        "documents_ingest: type=%s entities=%d confidence=%.2f",
        entry.kind, len(entry.entities), result.mean_confidence,
    )
    return entry


@app.post("/documents/timeline", response_model=DocumentBundle, tags=["documents"])
def documents_timeline(body: TimelineRequest) -> DocumentBundle:
    """A whole session's documents -> one dated, de-duplicated, abnormal-value-
    flagged DocumentBundle. **This is what the backend calls** once the patient
    has finished scanning; the result becomes the document half of the summary.
    """
    results: list[ExtractionResult] = []
    document_ids: list[str] = []
    source_texts: list[str] = []

    for doc in body.documents:
        source = _resolve_source(doc)
        result = structure(source, hint=doc.hint)
        result = code_result(result)
        build(result)
        results.append(result)
        document_ids.append(doc.document_id)
        source_texts.append(source.text)

    bundle = build_timeline(
        results, session_id=body.session_id, document_ids=document_ids, source_texts=source_texts,
    )
    log.info(  # counts only
        "documents_timeline: session=%s documents=%d timeline_entries=%d abnormal=%d",
        body.session_id, len(document_ids), len(bundle.timeline), len(bundle.abnormal_values),
    )
    return bundle


@app.post("/summary/generate", response_model=HistorySummary, tags=["summary"])
def summary_generate(body: SummaryGenerateRequest) -> HistorySummary:
    """Fuse the interview transcript and the document bundle into the physician's
    draft summary. **Always** `status="draft"` — the physician accepts, amends
    or rejects; nothing here is phrased as a diagnosis.
    """
    summary = build_history_summary(body.transcript, body.bundle)
    log.info(  # counts only
        "summary_generate: session=%s sections=%d contradictions=%d red_flags=%d",
        summary.session_id, len(summary.sections), len(summary.contradictions),
        len(summary.red_flags),
    )
    return summary


@app.post("/interactions", tags=["interactions"])
def interactions(body: InteractionRequest) -> dict:
    """Check a patient's full active medication list, pairwise.

    Send the **whole** list, not just the new prescription. Catching the pair no single
    prescriber could see is the entire argument for a longitudinal record.
    """
    alerts = check(body.drugs)
    return {
        "alerts": [a.model_dump() for a in alerts],
        "summary": summarize(alerts),
        #: Never render this result as "safe" — see checker.DISCLAIMER.
        "checked": len(body.drugs),
    }


@app.post("/admin/reload-terminology", tags=["ops"])
def reload_terminology() -> dict:
    """Pick up edits to `data/*.csv` without a restart. Dev convenience."""
    return {"terms": reload_tables()}


# ===========================================================================
# Module A — the interview. The backend's only way into the dialogue engine.
# ===========================================================================

#: One engine for the process. The ontology is parsed once at first use rather than
#: per request — it is a 30KB file and re-reading it per answer would put file I/O
#: in the loop a patient is waiting on.
_ENGINE: DialogueEngine | None = None

#: One lock per session. Sync FastAPI endpoints run in a threadpool, so a double-tap
#: on the kiosk or a client retry can enter this endpoint twice for the same session
#: concurrently and interleave reads and writes of the engine's per-session state.
_SESSION_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(session_id, threading.Lock())

#: LIMITATION, stated rather than hidden: interview state lives in this process's
#: memory. A restart mid-interview loses it, and a second replica would not share it.
#: Fine for one kiosk against one service; before running more than one instance this
#: needs the state moved into the backend's session store.
def _engine() -> DialogueEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DialogueEngine(
            str(ONTOLOGY_PATH),
            condition_map_path=str(CONDITION_MAP_PATH),
        )
    return _ENGINE


def _derive_ayush(answers: dict) -> dict:
    """The AYUSH parameters that are computed rather than asked.

    Two of the ten Dashavidha parameters are `derived_from` in the ontology and are
    therefore never put to the patient — asking someone their own prakriti would be
    asking them to do the examination. This is where they get filled in.

    Structural only. `missing` says which parameters have no answer, not whether any
    answer is clinically right; that judgement belongs to a practitioner and this
    module has no opinion about it.
    """
    from ..ayush import prakriti
    from ..ayush.dashavidha import derived_parameters, parameter_question_map
    from ..ayush.validator import AYUSHValidator

    result: dict = {}

    constitution = prakriti.derive(answers)
    if constitution is not None:
        result["prakriti"] = constitution

    mapping = parameter_question_map()
    validator = AYUSHValidator(answers)
    asked_missing = [
        p for p in
        validator.missing_dashavidha(mapping) + validator.missing_additional(mapping)
        # A derived parameter has no answer by construction. Counting it as missing
        # would report every complete interview as incomplete.
        if p not in derived_parameters()
    ]
    result["missing"] = asked_missing

    # vikriti and vaya are declared derived but nothing computes them yet - vaya needs
    # a date of birth the kiosk does not collect, and vikriti reads against prakriti.
    # Named rather than silently absent, so a physician sees the gap.
    result["not_derived"] = sorted(
        p for p in derived_parameters() if p not in result
    )
    result["complete"] = not asked_missing
    # The physician sees the gaps rather than a summary that implies a full
    # ten-fold examination was performed.
    return result


def _release_session(session_id: str) -> None:
    """Drop a finished interview's state and its lock."""
    engine = _engine()
    getattr(engine, "_sessions", {}).pop(session_id, None)
    with _LOCKS_GUARD:
        _SESSION_LOCKS.pop(session_id, None)


@app.post("/interview/next", response_model=InterviewNextOut, tags=["interview"])
def interview_next(body: InterviewNextIn) -> InterviewNextOut:
    """Record an answer and decide what to ask next. **The backend's endpoint.**

    `answer` is null to start. The engine walks the clinical ontology and applies
    the red-flag rules; a fired flag is REPORTED, never allowed to end the interview.
    """
    engine = _engine()

    with _session_lock(body.session_id):
        if body.answer is None or body.answer.question_id is None:
            question = engine.start_interview(
                session_id=body.session_id, language=body.language, mode=body.mode,
            )
        else:
            question = engine.submit_answer(body.session_id, body.answer)

        transcript = engine.get_transcript(body.session_id)

    # Counts only — never the answers themselves.
    log.info("interview: session step, completed=%s flags=%d",
             question is None, len(transcript.red_flags))

    if question is None:
        # Derived Dashavidha parameters, at the only point they can be computed: the
        # answers they read are spread across the whole interview. Everything here is
        # structural — nothing interprets an answer clinically.
        if transcript.mode == "ayush":
            transcript.ayush = _derive_ayush(transcript.answers)

        # The transcript is returned to the backend in this response and stored there,
        # so holding the session here after completion is pure memory growth — a kiosk
        # running all day would accumulate every patient it ever saw.
        _release_session(body.session_id)

    return InterviewNextOut(
        completed=question is None,
        question=question,
        progress={
            "answered": len(transcript.answers),
            "estimated_total": question.estimated_total if question else len(transcript.answers),
        },
        red_flags=transcript.red_flags,
        transcript=transcript if question is None else None,
    )
