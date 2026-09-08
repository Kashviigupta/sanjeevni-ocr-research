r"""Smoke-test the Bhashini inference key — MAHEK's setup tool.

Answers exactly one question: **does my INFERENCE key work against the dhruva
pipeline endpoint?** — separating auth problems from config problems, which the
raw curl error never does.

Usage (PowerShell):
    $env:BHASHINI_INFERENCE_KEY = "<paste key>"     # or put it in ml/.env
    python ml/scripts/test_bhashini.py              # 1s of silence, proves auth
    python ml/scripts/test_bhashini.py path\to\hindi.wav   # real transcription test

Reads BHASHINI_INFERENCE_KEY from the environment or ml/.env. Never prints the
key, never logs audio content. Exit code 0 = key accepted.
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

DHRUVA_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

#: Hindi conformer ASR — the serviceId used across Bhashini's own examples. If it
#: has been renamed, the API answers with a service error AFTER accepting the key,
#: which still proves the auth flow — the point of this script.
DEFAULT_SERVICE_ID = "ai4bharat/conformer-hi-gpu--t4"


def _load_key() -> str | None:
    key = os.environ.get("BHASHINI_INFERENCE_KEY")
    if key:
        return key.strip()
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("BHASHINI_INFERENCE_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A valid 16 kHz mono 16-bit WAV of near-silence, generated in-process.

    Good enough to authenticate and exercise the pipeline; obviously it should
    transcribe to nothing. Pass a real recording as argv[1] for a proper test.
    """
    n = int(seconds * rate)
    # A whisper-quiet 100 Hz hum instead of digital zero — some VADs reject
    # pure zeros as "no audio" before ASR ever runs.
    frames = b"".join(
        struct.pack("<h", int(80 * math.sin(2 * math.pi * 100 * i / rate))) for i in range(n)
    )
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(frames), b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * 2, 2, 16, b"data", len(frames),
    )
    return header + frames


def main() -> int:
    key = _load_key()
    if not key:
        print("FAIL  no key. Set BHASHINI_INFERENCE_KEY in the environment or ml/.env")
        return 2

    if len(sys.argv) > 1:
        audio = Path(sys.argv[1]).read_bytes()
        print(f"using audio file: {sys.argv[1]} ({len(audio)} bytes)")
    else:
        audio = _silence_wav()
        print("using generated near-silence (auth test only — pass a .wav for a real one)")

    body = {
        "pipelineTasks": [{
            "taskType": "asr",
            "config": {
                "language": {"sourceLanguage": "hi"},
                "serviceId": DEFAULT_SERVICE_ID,
                "audioFormat": "wav",
                "samplingRate": 16000,
            },
        }],
        "inputData": {"audio": [{"audioContent": base64.b64encode(audio).decode()}]},
    }

    request = urllib.request.Request(
        DHRUVA_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": key, "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as err:
        detail = err.read()[:300].decode(errors="replace")
        if err.code in (401, 403):
            print(f"FAIL  key REJECTED (HTTP {err.code}). Wrong key, or this key type "
                  "does not go direct-to-dhruva — fall back to the ULCA userID+ulcaApiKey flow.")
        else:
            print(f"OK-ISH  key ACCEPTED, request refused (HTTP {err.code}): {detail}\n"
                  "Auth works. Fix the config (serviceId/audio format) — see the response above.")
            return 0
        return 1
    except urllib.error.URLError as err:
        print(f"FAIL  network: {err.reason}")
        return 1

    text = ""
    for task in payload.get("pipelineResponse", []):
        if task.get("taskType") == "asr":
            outputs = task.get("output") or [{}]
            text = outputs[0].get("source", "")
    print("OK    key accepted, pipeline ran.")
    print(f'      transcript: "{text}"' if text else
          "      empty transcript (expected for the silence test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
