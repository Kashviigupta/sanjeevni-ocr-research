"""Tesseract OCR, with an optional vision-model fallback for handwriting.

Produces the `OcrResult` that extraction code consumes. Two hard requirements:

1. Never raise because the environment is missing something. A missing Tesseract binary
   returns an empty result with a warning, so the rest of the team keeps working.
2. Always report honest per-word confidence. The frontend highlights low-confidence
   fields for the patient to check, and that only works if the numbers mean something.
"""
from __future__ import annotations

import base64
import logging
from functools import lru_cache

from PIL import Image

from ..config import settings
from ..schemas import OcrResult, OcrWord
from .preprocess import preprocess

log = logging.getLogger(__name__)


def _tesseract():
    """Import lazily so the service starts without the binary installed."""
    import pytesseract

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    return pytesseract


def tesseract_version() -> str | None:
    try:
        return str(_tesseract().get_tesseract_version())
    except Exception:  # noqa: BLE001 - probing availability, not doing work
        return None


def available() -> bool:
    return tesseract_version() is not None


def _pdf_to_images(data: bytes) -> list[Image.Image]:
    """Rasterise a PDF with PyMuPDF — no system dependency, unlike pdf2image/poppler."""
    import fitz  # PyMuPDF

    pages: list[Image.Image] = []
    zoom = settings.pdf_dpi / 72
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc[: settings.max_pages]:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return pages


def _reconstruct_lines(data_dict: dict, page_no: int) -> tuple[list[OcrWord], str]:
    """Group Tesseract's flat word list back into the lines they came from.

    Tesseract already tells us which line each word belongs to via
    (block_num, par_num, line_num) — reading order within a page. Losing that and
    joining every word with a single space collapses a lab report's rows into one
    run-on line, which is exactly the row structure `extraction.patterns.LAB_ROW_RE`
    needs to find an analyte/value/unit triple. This function is the fix: it walks
    the words in the order Tesseract already put them in, and inserts a newline
    whenever the (block, paragraph, line) key changes.
    """
    words: list[OcrWord] = []
    lines: list[list[str]] = []
    current_key: tuple[int, int, int] | None = None
    current_line: list[str] = []

    for i, raw in enumerate(data_dict["text"]):
        token = raw.strip()
        if not token:
            continue
        #: Tesseract reports 0-100, and -1 for non-text blocks.
        conf = float(data_dict["conf"][i])
        if conf < 0:
            continue

        key = (data_dict["block_num"][i], data_dict["par_num"][i], data_dict["line_num"][i])
        if current_key is not None and key != current_key:
            lines.append(current_line)
            current_line = []
        current_key = key
        current_line.append(token)

        words.append(
            OcrWord(
                text=token,
                confidence=round(conf / 100.0, 4),
                bbox=(
                    int(data_dict["left"][i]),
                    int(data_dict["top"][i]),
                    int(data_dict["width"][i]),
                    int(data_dict["height"][i]),
                ),
                page=page_no,
            )
        )

    if current_line:
        lines.append(current_line)

    text = "\n".join(" ".join(line) for line in lines)
    return words, text


#: The rest of the platform speaks ISO 639-1 - a session's language is "hi" or "en",
#: and that value is handed straight down to here. Tesseract speaks ISO 639-2/T and
#: fails outright on a two-letter code: passing "en" raised TesseractError, which was
#: caught, logged, and returned as an empty page with 0.00 confidence. Every document
#: upload silently read nothing, on every language the product actually uses.
_ISO_639_1_TO_TESSERACT = {
    "en": "eng", "hi": "hin", "bn": "ben", "mr": "mar", "ta": "tam", "te": "tel",
    "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "or": "ori", "as": "asm",
    "ur": "urd", "ne": "nep", "sa": "san",
}


@lru_cache(maxsize=1)
def installed_languages() -> frozenset[str]:
    """Language packs actually present on this machine.

    `tesseract-ocr-hin` is a separate apt package. Asking for a pack that is not
    installed raises exactly the same TesseractError as a malformed code, and produces
    exactly the same silent empty page - so the default must be filtered against
    reality rather than assumed.
    """
    try:
        return frozenset(_tesseract().get_languages(config=""))
    except Exception:  # noqa: BLE001 - tesseract absent is handled by available()
        return frozenset()


def resolve_language(language: str | None) -> str:
    """An app locale to a Tesseract language string this machine can actually load.

    Already-Tesseract values pass through ("eng", "eng+hin"), so callers that know what
    they are doing keep working. Anything unknown, or any pack that is not installed,
    is dropped rather than passed on: reading a document in one script still gives a
    physician something to correct, while raising gives them an empty page.
    """
    parts = [p.strip() for p in (language or settings.ocr_languages).split("+") if p.strip()]
    mapped = [_ISO_639_1_TO_TESSERACT.get(p, p) for p in parts]
    mapped = [p for p in mapped if len(p) >= 3]
    present = installed_languages()
    if present:
        usable = [p for p in mapped if p in present]
        if usable:
            return "+".join(usable)
        # Nothing requested is installed. English is the realistic fallback for Indian
        # medical documents, which are overwhelmingly printed in English.
        return "eng" if "eng" in present else next(iter(sorted(present)))
    return "+".join(mapped) or settings.ocr_languages


def _line_confidences(data_dict: dict) -> list[float]:
    """Mean word confidence per (block, paragraph, line) group, for lines of 2+ words.

    Exists to catch a real failure mode: a printed letterhead sitting above a
    completely garbled handwritten drug list can average out to a page-level
    `mean_confidence` that looks fine (e.g. 0.66), because the clean printed
    words drag the overall average up while the handwriting underneath was
    never actually read. A single flat average has no way to see that — this
    does, by keeping every line's own confidence separate so the caller can
    check the *worst* line, not just the page-wide mean.

    Single-word lines are excluded on purpose: one garbled token (a stray
    mark, a stamp, a signature Tesseract misreads as a word) is noise, not
    evidence that an entire line needs the vision fallback — escalating on
    that would send documents to a hosted model for no real reason.
    """
    groups: dict[tuple[int, int, int], list[float]] = {}
    for i, raw in enumerate(data_dict["text"]):
        token = raw.strip()
        if not token:
            continue
        conf = float(data_dict["conf"][i])
        if conf < 0:
            continue
        key = (data_dict["block_num"][i], data_dict["par_num"][i], data_dict["line_num"][i])
        groups.setdefault(key, []).append(conf / 100.0)
    return [sum(confs) / len(confs) for confs in groups.values() if len(confs) >= 2]


def _run_ocr_with_line_stats(
    data: bytes, media_type: str = "image/jpeg", language: str | None = None,
    do_preprocess: bool = True,
) -> tuple[OcrResult, list[float]]:
    """Everything `run_ocr` does, plus the per-line confidences `ocr_document`
    needs to decide on vision escalation. Factored out so `run_ocr`'s public
    signature and behaviour stay exactly as every existing caller and test
    already expects — this is the only function that also tracks line-level
    detail, and only `ocr_document` uses that extra return value.
    """
    lang = resolve_language(language)

    if not available():
        return (
            OcrResult(
                text="",
                mean_confidence=0.0,
                engine="unavailable",
                language=lang,
                warnings=[
                    "tesseract binary not found - install it and set TESSERACT_CMD; "
                    "the app falls back to manual entry"
                ],
            ),
            [],
        )

    pytesseract = _tesseract()
    images = _pdf_to_images(data) if media_type == "application/pdf" else [None]

    all_words: list[OcrWord] = []
    page_texts: list[str] = []
    warnings: list[str] = []
    all_line_confidences: list[float] = []

    for page_no, page_img in enumerate(images):
        img = preprocess(data) if page_img is None and do_preprocess else page_img
        if img is None:
            from .preprocess import load_image

            img = load_image(data)

        try:
            data_dict = pytesseract.image_to_data(
                img, lang=lang, output_type=pytesseract.Output.DICT
            )
        except Exception as exc:  # noqa: BLE001
            # Never log the text itself. The language IS worth logging: a missing
            # language pack looks identical to an unreadable page from the outside.
            log.warning("ocr failed on page %s (lang=%s): %s", page_no, lang, exc)
            warnings.append(f"page {page_no}: OCR failed with lang={lang} ({type(exc).__name__})")
            continue

        page_words, page_text = _reconstruct_lines(data_dict, page_no)
        all_words.extend(page_words)
        page_texts.append(page_text)
        all_line_confidences.extend(_line_confidences(data_dict))

    mean_conf = round(sum(w.confidence for w in all_words) / len(all_words), 4) if all_words else 0.0
    if all_words and mean_conf < settings.vision_fallback_threshold:
        warnings.append(
            f"low OCR confidence ({mean_conf:.2f}) - likely handwriting or a poor photo"
        )

    result = OcrResult(
        text="\n\n".join(page_texts).strip(),
        words=all_words,
        mean_confidence=mean_conf,
        pages=len(images),
        engine=f"tesseract-{tesseract_version()}",
        language=lang,
        warnings=warnings,
    )
    return result, all_line_confidences


def run_ocr(data: bytes, media_type: str = "image/jpeg", language: str | None = None,
            do_preprocess: bool = True) -> OcrResult:
    """Document bytes to text plus per-word confidence.

    `media_type` decides whether we rasterise first. `language` is a Tesseract language
    string — "eng", "hin", or "eng+hin" for the mixed-script documents that are normal
    in Indian clinics.
    """
    result, _line_confidences = _run_ocr_with_line_stats(
        data, media_type=media_type, language=language, do_preprocess=do_preprocess
    )
    return result


def _run_vision_ocr_anthropic(data: bytes, media_type: str) -> OcrResult:
    """Anthropic Messages API vision transcription."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Transcribe every word of text visible in this image exactly "
                        "as written. Plain transcription only - no interpretation, no "
                        "dosage reasoning, no corrections, no summary. If handwriting "
                        "is illegible, write [illegible] in its place rather than "
                        "guessing."
                    ),
                },
            ],
        }],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return OcrResult(
        text=text.strip(),
        words=[],  # a vision model gives no per-word bounding boxes or scores
        #: Never fabricate a high number: a vision model has no calibrated per-word
        #: confidence to report, so this is a deliberately conservative flat estimate,
        #: not a measurement. Kept below Tesseract's own escalation threshold so a
        #: vision result never masquerades as more certain than it actually is.
        mean_confidence=min(0.6, settings.vision_fallback_threshold),
        engine=f"claude-vision-{settings.anthropic_model}",
        language="auto",
    )


def _run_vision_ocr_gemini(data: bytes, media_type: str) -> OcrResult:
    """Gemini vision transcription — the fallback actually configured for this repo.

    Uses the current `google-genai` SDK, not the deprecated `google-generativeai`
    package (which Google has publicly end-of-lifed) — no reason to build new code
    against a package that's already announced as unsupported.
    """
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    last_exc = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[
                    genai.types.Part.from_bytes(data=data, mime_type=media_type),
                    "Transcribe every word of text visible in this image exactly as written. "
                    "Plain transcription only - no interpretation, no dosage reasoning, no "
                    "corrections, no summary. If handwriting is illegible, write [illegible] "
                    "in its place rather than guessing.",
                ],
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            is_503 = "503" in msg or "UNAVAILABLE" in msg
            # A 429 is only worth retrying when it's a transient per-minute rate
            # limit -- retrying a daily quota error (the common free-tier case)
            # just wastes the retry budget sleeping on a cap that won't lift
            # until tomorrow, so check for the daily-limit wording explicitly.
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            is_daily_quota = is_429 and "PerDay" in msg
            if (is_503 or (is_429 and not is_daily_quota)) and attempt < 2:
                import time
                time.sleep(2 * (attempt + 1))  # 2s, then 4s
                continue
            raise
    text = (response.text or "").strip()
    return OcrResult(
        text=text,
        words=[],  # a vision model gives no per-word bounding boxes or scores
        #: Same conservative-estimate rule as the Anthropic path above — never a
        #: fabricated high number for a model with no real per-word calibration.
        mean_confidence=min(0.6, settings.vision_fallback_threshold),
        engine=f"gemini-vision-{settings.gemini_model}",
        language="auto",
    )


def run_vision_ocr(data: bytes, media_type: str = "image/jpeg") -> OcrResult | None:
    """Optional: a vision model reads handwriting far better than Tesseract does.

    Returns None when no vision provider is configured — callers must handle that
    and fall back to `run_ocr`. **The offline path is the product; this is a bonus.**
    Provider choice is `settings.vision_provider`: self-hosted MedGemma when one is
    configured, otherwise a hosted provider — and only with `ALLOW_HOSTED_VISION`,
    because the hosted free tiers train on what they are sent.
    """
    provider = settings.vision_provider
    if provider in (None, "self_hosted"):
        # MedGemma is not wired yet. Returning None keeps the promise in this
        # docstring — the offline path is the product — instead of silently
        # shipping the document to a third party because the good option is missing.
        return None
    if provider == "anthropic":
        return _run_vision_ocr_anthropic(data, media_type)
    return _run_vision_ocr_gemini(data, media_type)


def ocr_document(data: bytes, media_type: str = "image/jpeg",
                 language: str | None = None) -> OcrResult:
    """The entry point the service calls: Tesseract first, vision model only if poor.

    Escalation checks TWO signals, not one: the page's overall mean confidence
    (the original check), and the worst individual line's confidence (see
    `_line_confidences`). A document can have a fine overall average while one
    genuinely important line — the handwritten drug list under a clean printed
    letterhead — was barely read at all. Checking only the average misses
    exactly that case; checking the worst line catches it.
    """
    result, line_confidences = _run_ocr_with_line_stats(data, media_type=media_type, language=language)

    worst_line_confidence = min(line_confidences) if line_confidences else result.mean_confidence
    needs_escalation = (
        settings.vision_enabled
        and (
            result.mean_confidence < settings.vision_fallback_threshold
            or worst_line_confidence < settings.vision_fallback_threshold
        )
    )

    if needs_escalation:
        try:
            better = run_vision_ocr(data, media_type=media_type)
            if better is not None:
                reason = (
                    f"tesseract confidence was {result.mean_confidence:.2f}"
                    if result.mean_confidence < settings.vision_fallback_threshold
                    else f"a line's confidence was {worst_line_confidence:.2f} despite a "
                         f"{result.mean_confidence:.2f} page average — likely handwriting "
                         f"under a printed letterhead"
                )
                better.warnings.append(f"escalated to vision model ({reason})")
                return better
        except Exception:  # noqa: BLE001 - a failed vision call must never crash the request
            log.warning("vision fallback failed; keeping the tesseract result", exc_info=True)

    return result
