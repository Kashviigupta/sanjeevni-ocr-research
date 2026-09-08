"""Text-to-speech via Bhashini TTS.

Every prompt the dialogue engine sends to a patient must be spoken aloud --
this is the only place that turns question text into audio for that.

NOTE: unlike ASR, TTS has not yet been smoke-tested against our real key
(ml/scripts/test_bhashini.py only checked ASR). Confirm the service IDs in
bhashini_client.TTS_SERVICE_IDS work before relying on this in a demo.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from .bhashini_client import BhashiniError, TTS_SERVICE_IDS, compute


@dataclass
class TtsResult:
    audio_bytes: bytes
    language: str


def synthesize(text: str, language: str, gender: str = "female") -> TtsResult:
    """Turn `text` into spoken audio in `language`.

    `language` must be an ISO-639 code with a known service ID (see
    TTS_SERVICE_IDS -- currently unconfirmed against our real key).
    """
    if language not in TTS_SERVICE_IDS:
        raise ValueError(
            f"No known TTS service ID for language '{language}'. "
            f"Supported so far: {list(TTS_SERVICE_IDS)}"
        )

    payload = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": language},
                    "serviceId": TTS_SERVICE_IDS[language],
                    "gender": gender,
                    "samplingRate": 48000,
                },
            }
        ],
        "inputData": {
            "input": [{"source": text}],
            "audio": [{"audioContent": None}],
        },
    }

    response = compute(payload)

    try:
        audio_content_b64 = response["pipelineResponse"][0]["audio"][0]["audioContent"]
    except (KeyError, IndexError) as exc:
        raise BhashiniError(
            f"Unexpected TTS response shape from Bhashini: {response}"
        ) from exc

    return TtsResult(
        audio_bytes=base64.b64decode(audio_content_b64),
        language=language,
    )
