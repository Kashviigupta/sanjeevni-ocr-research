"""Speech-to-text via Bhashini ASR.

This is the ONLY place that talks to Bhashini for ASR. The dialogue engine
never calls Bhashini directly -- it calls transcribe() and gets back
structured text + confidence, so it can decide whether to trust it or ask
the patient to confirm (per the "honest confidence" rule: never silently
record a low-confidence guess).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from .bhashini_client import ASR_SERVICE_IDS, BhashiniError, compute


@dataclass
class AsrResult:
    """What the dialogue engine actually needs from a transcription.

    confidence is None when Bhashini's response doesn't include a score for
    this model -- callers must treat None as "unknown", not as 1.0.
    """

    text: str
    confidence: float | None
    language: str


def transcribe(audio_bytes: bytes, language: str, audio_format: str = "wav",
               sampling_rate: int = 16000) -> AsrResult:
    """Send raw audio bytes to Bhashini ASR and return the transcript.

    `language` must be an ISO-639 code with a known service ID (currently
    "hi" -- confirmed -- or "en" -- unconfirmed. See ASR_SERVICE_IDS).
    Extending to Marathi/Bengali/Tamil/Telugu means confirming their real
    service IDs the same way "hi" was confirmed, not guessing from docs.
    """
    if language not in ASR_SERVICE_IDS:
        raise ValueError(
            f"No known ASR service ID for language '{language}'. "
            f"Supported so far: {list(ASR_SERVICE_IDS)}"
        )

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": language},
                    "serviceId": ASR_SERVICE_IDS[language],
                    "audioFormat": audio_format,
                    "samplingRate": sampling_rate,
                },
            }
        ],
        "inputData": {
            # An empty string, not None. Bhashini's schema rejects null here with a
            # 422 on inputData.input[0].source - ASR has no source text by definition,
            # but the field still has to be present and a string.
            "input": [{"source": ""}],
            "audio": [{"audioContent": base64.b64encode(audio_bytes).decode("ascii")}],
        },
    }

    response = compute(payload)

    try:
        asr_output = response["pipelineResponse"][0]["output"][0]
    except (KeyError, IndexError) as exc:
        raise BhashiniError(
            f"Unexpected ASR response shape from Bhashini: {response}"
        ) from exc

    return AsrResult(
        text=asr_output.get("source", ""),
        # Bhashini doesn't always return a confidence score for ASR; when it
        # doesn't, we surface None rather than inventing a number.
        confidence=asr_output.get("confidence"),
        language=language,
    )
