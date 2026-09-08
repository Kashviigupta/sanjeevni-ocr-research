"""The shapes the two halves of this module agree on.

`OcrResult` is the seam: Mahek's OCR code produces it, Kashvi's extraction code
consumes it. Neither half needs to read the other's implementation, and either can
be developed against a hand-written fixture while the other is still in progress.

These mirror the ML section of `shared/types/api.ts`. Change one, change both.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- enums

RecordType = Literal["condition", "prescription", "lab", "allergy", "imaging", "note"]
EntityKind = Literal["lab", "medication", "diagnosis", "vital", "date", "facility"]
CodeSystem = Literal["LOINC", "SNOMED", "ICD-11", "RxNorm", "NAMASTE"]
Severity = Literal["minor", "moderate", "major", "contraindicated"]
MediaType = Literal["image/jpeg", "image/png", "image/webp", "application/pdf"]


# ----------------------------------------------------------------------- OCR (Mahek)

class OcrWord(BaseModel):
    """One recognised word with its own confidence.

    Per-word confidence — rather than one number for the page — is what lets the
    frontend highlight exactly which fields the patient should double-check.
    """

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels of the preprocessed image
    page: int = 0


class OcrResult(BaseModel):
    """THE HANDOFF. Mahek produces this; Kashvi consumes it.

    Do not change this shape without telling the other half of the module.
    """

    text: str
    words: list[OcrWord] = Field(default_factory=list)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    pages: int = 1
    #: Provenance — e.g. "tesseract-5.3.0" or "claude-vision". A clinician must always
    #: be able to ask which engine produced a number.
    engine: str = "unknown"
    #: Detected script(s): "eng", "hin", or "eng+hin" for the mixed documents that are
    #: normal in Indian clinics.
    language: str = "eng"
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------- extraction (Kashvi)

class TerminologyCode(BaseModel):
    """A real code from a real table. Never fabricate one to fill this field."""

    system: CodeSystem
    code: str
    display: str


class ExtractedEntity(BaseModel):
    """One clinical fact pulled out of a document, before it becomes FHIR."""

    kind: EntityKind
    text: str                                   # exactly as it appeared in the document
    value: str | float | None = None
    unit: str | None = None
    reference_range: str | None = None
    #: None when no curated table entry matched. That is a correct answer, not a failure.
    code: TerminologyCode | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """What the backend receives, and what the patient is asked to verify.

    Everything here is a *proposal*. Nothing in this object is clinical truth until a
    human has confirmed it in the app.
    """

    entities: list[ExtractedEntity] = Field(default_factory=list)
    suggested_type: RecordType = "note"
    suggested_title: str = "Untitled document"
    clinical_date: str | None = None            # YYYY-MM-DD
    facility: str | None = None
    #: A FHIR R4 resource — Observation, MedicationRequest, DiagnosticReport, Condition.
    fhir: dict[str, Any] = Field(default_factory=dict)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    engine: str = "unknown"
    warnings: list[str] = Field(default_factory=list)


class InteractionAlert(BaseModel):
    """A drug-drug interaction. Severity drives how loudly the UI shouts."""

    severity: Severity
    drug_a: str
    drug_b: str
    description: str
    source: str = "curated-table-v1"


# ------------------------------------------------------------------ request bodies

class OcrRequest(BaseModel):
    data_base64: str = Field(min_length=16, max_length=14_000_000)  # ~10 MB decoded
    media_type: MediaType = "image/jpeg"
    #: None uses the configured scripts. Accepts an app locale ("hi") or a Tesseract
    #: string ("eng+hin") — see ocr.engine.resolve_language.
    language: str | None = None
    preprocess: bool = True


class ExtractRequest(BaseModel):
    """Either hand us raw text, or a document to OCR first."""

    text: str | None = None
    data_base64: str | None = None
    media_type: MediaType = "image/jpeg"
    #: The language of the PAPER, which is rarely the language of the patient. None
    #: means "nobody knows", and the service falls back to its configured scripts -
    #: the honest answer when a document arrives at a kiosk unannounced.
    language: str | None = None
    hint: RecordType | None = None  # the patient may say "this is a lab report"


class InteractionRequest(BaseModel):
    #: Drug names as written — brand or generic. The checker normalises them.
    drugs: list[str] = Field(min_length=1, max_length=50)

class DocumentIngestRequest(BaseModel):
    """One document in, one TimelineEntry out. Same source rules as ExtractRequest:
    either hand over text directly, or a document to OCR first.
    """

    document_id: str
    text: str | None = None
    data_base64: str | None = None
    media_type: MediaType = "image/jpeg"
    #: The language of the PAPER. The backend deliberately does not pass the session's
    #: locale here: an Indian patient speaking Hindi hands over a prescription printed
    #: in English, and sending "hi" made Tesseract read it with the Devanagari model
    #: and return nothing. None means the service uses its configured scripts.
    language: str | None = None
    hint: RecordType | None = None


class TimelineRequest(BaseModel):
    """A whole session's documents, to be OCR'd/extracted/coded and assembled
    into one dated, de-duplicated, abnormal-value-flagged DocumentBundle.
    """

    session_id: str
    documents: list[DocumentIngestRequest] = Field(min_length=1, max_length=50)


class SummaryGenerateRequest(BaseModel):
    """Both halves Module C needs: Mahek's transcript and this service's own bundle."""

    transcript: InterviewTranscript
    bundle: DocumentBundle


# ------------------------------------------------------------------------- speech
# Voice is the whole accessibility argument: a patient who cannot read must be able to
# finish the interview by ear and by speaking. Both of these wrap Bhashini, which is
# free and Indian, and both degrade rather than fail — a kiosk that loses its voice
# falls back to touch, it does not stop.

class AsrRequest(BaseModel):
    """Spoken answer in, text out."""

    data_base64: str = Field(min_length=16, max_length=14_000_000)
    #: ISO-639-1 as the rest of the platform speaks it: "hi", "en".
    language: str = "hi"
    audio_format: Literal["wav", "mp3", "flac", "ogg"] = "wav"


class AsrResponse(BaseModel):
    text: str
    #: None means Bhashini returned no score for this model. Treat it as *unknown* -
    #: never as 1.0. A confidently-wrong transcription of a symptom is worse than an
    #: admitted uncertainty a clinician can check.
    confidence: float | None = None
    language: str


class TtsRequest(BaseModel):
    """Prompt text in, spoken audio out."""

    text: str = Field(min_length=1, max_length=2000)
    language: str = "hi"
    gender: Literal["female", "male"] = "female"


class TtsResponse(BaseModel):
    audio_base64: str
    media_type: str = "audio/wav"
    language: str


class ServiceStatus(BaseModel):
    status: Literal["ok", "degraded"]
    tesseract: bool
    tesseract_version: str | None = None
    vision_llm: bool
    terminology_terms: int
    detail: str | None = None


# ===========================================================================
# Module A — the conversation (MAHEK)
#
# The dialogue engine walks `shared/clinical/history-ontology.json` and returns
# one Question at a time. Clients render; they never decide what to ask next.
# ===========================================================================

InputType = Literal[
    "single_choice", "multi_choice", "scale", "body_map",
    "duration", "yes_no", "free_speech", "number", "date",
]
RedFlagSeverity = Literal["critical", "urgent", "watch"]
InterviewMode = Literal["allopathic", "ayush"]


class QuestionOption(BaseModel):
    """One tappable tile. `icon` carries the meaning; `label` is the backup."""

    id: str
    label: dict[str, str]              # language code -> text, for every supported language
    icon: str | None = None
    exclusive: bool = False            # e.g. "None of these" clears the rest


class Question(BaseModel):
    """What a client renders. Everything needed to draw one screen.

    `audio_url` is not optional in practice: every prompt is spoken aloud, because a
    patient who cannot read must be able to finish the interview by ear alone.
    """

    question_id: str
    section_id: str
    input_type: InputType
    prompt: dict[str, str]             # language code -> text
    audio_url: str | None = None
    options: list[QuestionOption] = Field(default_factory=list)
    skippable: bool = False
    #: Rendering hints the type needs — scale min/max, body-map regions, duration units.
    hints: dict[str, Any] = Field(default_factory=dict)
    #: For a progress bar. `estimated_total` moves as the interview branches, so the UI
    #: should say "about" rather than implying a fixed length.
    answered: int = 0
    estimated_total: int = 0
    #: Why the engine chose THIS question now. Never shown to the patient — it exists so
    #: we can explain the interview to a judge and debug a bad question order.
    selection_reason: str | None = None


class DurationValue(BaseModel):
    """A `duration` answer: "three days", not a free-text guess at what that means.

    The unit vocabulary is closed and matches `redflags.rules._HOURS_PER_UNIT` exactly,
    because red flags compare against `value_hours`. `soc.onset` fires
    `chest_pain_acute_onset` at one hour - a client inventing "day"/"dy"/"days " makes
    that comparison silently wrong on the most time-critical question we ask.
    """

    amount: float = Field(ge=0)
    unit: Literal["hours", "days", "weeks", "months", "years"]


class Answer(BaseModel):
    """What a client sends back. Exactly one of `value` or `audio_base64`.

    `transcript` and `asr_confidence` are filled in by the speech layer, not the client.
    """

    question_id: str
    value: str | float | list[str] | DurationValue | None = None
    audio_base64: str | None = None
    transcript: str | None = None
    asr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str = "hi"
    #: True when the patient tapped rather than spoke. Worth recording: it tells us
    #: which questions the voice path is failing on.
    via_touch: bool = True


class RedFlag(BaseModel):
    """An emergency signal routed to triage staff.

    NEVER surfaced to the patient as an emergency — that is a diagnosis, and we do not
    diagnose. The kiosk says a staff member will come to them shortly.
    """

    reason: str                        # machine code, e.g. "chest_pain_with_dyspnoea"
    severity: RedFlagSeverity
    question_id: str
    #: Physician-facing English/Hindi text. Never rendered on the kiosk.
    explanation: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------- disease-correlated questioning

class ConditionHypothesis(BaseModel):
    """A condition the engine is currently exploring — INTERNAL, never a diagnosis.

    This is the mechanism that makes the interview correlated rather than a flat
    questionnaire: answers raise and lower plausibility, and the next question is
    chosen to best discriminate between whatever is still live.

    **It is a question-selection device, not a clinical conclusion.** The PS forbids
    autonomous diagnosis, so this object must never reach the patient, and reaches the
    physician only as "what was explored" — never as "what the patient has".
    """

    condition_id: str
    label: dict[str, str]
    code: TerminologyCode | None = None
    #: 0-1. Deliberately NOT called "probability" — it is an unvalidated heuristic
    #: score for ordering questions, and calling it a probability invites someone to
    #: display it as one.
    plausibility: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Answers that raised it, so a physician can see the engine's reasoning.
    supported_by: list[str] = Field(default_factory=list)
    #: Answers that lowered it.
    contradicted_by: list[str] = Field(default_factory=list)
    urgency: RedFlagSeverity | None = None


class ExplorationReport(BaseModel):
    """What the interview covered, for the physician's eyes only.

    The honest version of "here is what the AI was thinking": which areas were probed,
    which were ruled out by an explicit answer, and — most importantly — which were
    left unexplored because the patient stopped or the engine ran out of budget.
    """

    explored: list[ConditionHypothesis] = Field(default_factory=list)
    #: Lowered by an explicit negative answer. Useful: "no chest pain on exertion" is
    #: clinical information, not an absence of information.
    ruled_down: list[ConditionHypothesis] = Field(default_factory=list)
    #: Areas a complete history would have covered and this interview did not.
    #: Stating the gap is the difference between a useful summary and a misleading one.
    not_explored: list[str] = Field(default_factory=list)


class InterviewTranscript(BaseModel):
    """THE HANDOFF from Mahek to Kashvi. Module C's summary generator consumes this.

    Do not change this shape without telling the other half of the module.
    """

    session_id: str
    mode: InterviewMode = "allopathic"
    language: str = "hi"
    #: question_id -> the structured answer, already normalised by the dialogue engine.
    answers: dict[str, Any] = Field(default_factory=dict)
    #: The patient's own words, verbatim. The most clinically valuable content in the
    #: whole interview — never paraphrase it away.
    narration: list[str] = Field(default_factory=list)
    red_flags: list[RedFlag] = Field(default_factory=list)
    exploration: ExplorationReport = Field(default_factory=ExplorationReport)
    completed: bool = False
    #: Questions skipped or unresolved. The physician should see the gaps, not a summary
    #: that pretends the history is complete.
    unanswered: list[str] = Field(default_factory=list)
    engine: str = "dialogue-v1"
    #: AYUSH mode only. The Dashavidha parameters that are computed rather than asked -
    #: prakriti from its sub-questionnaire, vaya from date of birth - plus a
    #: completeness check. Empty for an allopathic interview, and empty in AYUSH mode
    #: when too little was answered to derive anything: a constitution read off two
    #: questions is not a constitution.
    ayush: dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# Modules B + C — documents and the summary (KASHVI)
# ===========================================================================

class TimelineEntry(BaseModel):
    """One dated event in the patient's document history."""

    date: str | None = None            # YYYY-MM-DD; None when undateable
    date_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    kind: RecordType
    title: str
    facility: str | None = None
    document_id: str | None = None     # the original scan, for side-by-side review
    #: Other scans of the SAME document, folded into this one. The physician can still
    #: open every page the patient handed over — de-duplication shortens the timeline,
    #: it never hides a record.
    duplicate_of: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    fhir: dict[str, Any] = Field(default_factory=dict)


class AbnormalValue(BaseModel):
    """An out-of-range lab result, flagged for physician attention.

    The direction and the range are reported; no interpretation is offered. We do not
    say what an abnormal value means — that is the physician's job.
    """

    analyte: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    direction: Literal["high", "low"]
    code: TerminologyCode | None = None
    source_document_id: str | None = None


class DocumentBundle(BaseModel):
    """Everything Module B extracted from the patient's paper, in date order."""

    session_id: str
    timeline: list[TimelineEntry] = Field(default_factory=list)
    abnormal_values: list[AbnormalValue] = Field(default_factory=list)
    interactions: list[InteractionAlert] = Field(default_factory=list)
    documents_processed: int = 0
    #: Documents we could not read at all. Show the patient and offer a retake — never
    #: silently drop a record they handed us.
    unreadable: list[str] = Field(default_factory=list)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Ambiguous dates, merge decisions, and capabilities that are not implemented.
    warnings: list[str] = Field(default_factory=list)


class SummarySection(BaseModel):
    """One block of the physician-ready summary.

    `provenance` is what makes a claim answerable. Every line must trace to the
    patient's own words or to a named document — a physician has to be able to ask
    "how do you know that?" and get an answer.
    """

    heading: Literal[
        "chief_complaint", "hpi", "past_medical", "past_surgical",
        "drugs_allergy", "family", "personal", "ros",
        "prior_investigations", "ayurvedic_assessment",
    ]
    content: dict[str, str]            # language code -> physician-facing text
    provenance: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class Contradiction(BaseModel):
    """The interview and a document disagree.

    SURFACE both; never resolve silently. "Patient reports no medicines" against a
    scanned prescription for metformin is exactly what the physician must see, and
    exactly what an eager summariser would smooth over.
    """

    topic: str
    from_interview: str
    from_documents: str
    document_id: str | None = None


class HistorySummary(BaseModel):
    """MODULE C OUTPUT — the artefact the whole project is judged on.

    Always a draft. The physician accepts, amends or rejects. The PS is explicit:
    never an autonomous diagnosis. Do not add a differential, a likely cause or a
    recommendation to this object, however tempting it is.
    """

    session_id: str
    sections: list[SummarySection] = Field(default_factory=list)
    red_flags: list[RedFlag] = Field(default_factory=list)
    abnormal_values: list[AbnormalValue] = Field(default_factory=list)
    interactions: list[InteractionAlert] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    exploration: ExplorationReport = Field(default_factory=ExplorationReport)
    gaps: list[str] = Field(default_factory=list)
    patient_confirmation: dict[str, str] = Field(default_factory=dict)
    fhir_bundle: dict[str, Any] = Field(default_factory=dict)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    engine: str = "summary-v1"
    status: Literal["draft"] = "draft"


class InterviewNextIn(BaseModel):
    """The backend asking what to put to the patient next. `answer` null starts."""

    session_id: str
    mode: InterviewMode = "allopathic"
    language: str = "hi"
    answer: Answer | None = None


class InterviewNextOut(BaseModel):
    """The next question, or the finished transcript.

    `question` is None exactly when `completed` is true — clients branch on that
    rather than on an empty object. `transcript` is returned only at the end, so an
    in-progress interview never ships the whole answer set over the wire repeatedly.
    """

    completed: bool = False
    question: Question | None = None
    progress: dict[str, int] = Field(default_factory=dict)
    #: Physician-facing. The kiosk is told only THAT one fired, never what it means.
    red_flags: list[RedFlag] = Field(default_factory=list)
    transcript: InterviewTranscript | None = None
