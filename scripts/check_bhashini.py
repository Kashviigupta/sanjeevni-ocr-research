"""Bare-minimum Bhashini connectivity check.

Run this BEFORE building anything on top of the ASR/TTS wrappers. It does
one thing: sends one small audio clip to Bhashini and prints exactly what
comes back -- success or a real error message. No wrapper code, no
abstractions, nothing to debug except "does my key work."

Not a test - a one-shot connectivity probe that makes a real network call and needs a
real key. It was named `test_connection.py` at the package root, where the name implies
pytest ownership it does not have.

Usage:
    pip install requests python-dotenv
    python scripts/check_bhashini.py

Expects a .env file (same folder or a parent folder) containing:
    BHASHINI_INFERENCE_KEY=your-key-here
"""

import base64
import math
import struct
import wave
import io
import os

import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
HINDI_ASR_SERVICE_ID = "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4"


def make_test_wav_bytes() -> bytes:
    """Generate one second of a simple tone as a valid 16kHz mono WAV.

    This is NOT meant to test transcription accuracy -- it's silence/a
    tone, so Bhashini will likely return an empty or nonsense transcript.
    The point of this script is only to confirm the connection, auth, and
    request shape are correct. Real accented Hindi audio fixtures come later.
    """
    sample_rate = 16000
    duration_seconds = 1
    frequency = 440  # A4 tone, arbitrary

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(sample_rate * duration_seconds):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack("<h", value))
    return buffer.getvalue()


def main() -> None:
    key = os.environ.get("BHASHINI_INFERENCE_KEY")
    if not key:
        print("BHASHINI_INFERENCE_KEY not found. Check your .env file exists and has this exact variable name.")
        return

    audio_bytes = make_test_wav_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": "hi"},
                    "serviceId": HINDI_ASR_SERVICE_ID,
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                },
            }
        ],
        "inputData": {
            "input": [{"source": None}],
            "audio": [{"audioContent": audio_b64}],
        },
    }

    print("Sending request to Bhashini...")
    response = requests.post(
        ENDPOINT,
        headers={"Authorization": key, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )

    print(f"\nStatus code: {response.status_code}")
    print("Response body:")
    print(response.text)

    if response.status_code == 200:
        print("\n✅ Connection works. Your key and endpoint are correctly set up.")
    else:
        print("\n❌ Something's wrong. Copy the status code and response body above and send it to Claude.")


if __name__ == "__main__":
    main()
