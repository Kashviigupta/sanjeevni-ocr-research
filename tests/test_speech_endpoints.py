"""The two endpoints that make voice reachable.

Mahek's ASR and TTS clients both worked, and neither had an HTTP surface — so the
kiosk could not speak a prompt or hear an answer, and the headline demo ("a patient
completes the whole interview by voice, in Hindi, without touching the screen") was
unshootable.

These tests do not call Bhashini. They cover the contract and, more importantly, the
degradation: a kiosk that loses its voice must fall back to touch, never stop.
"""
import base64

import pytest
from fastapi.testclient import TestClient

from sanjeevani_ml.service.main import app
from sanjeevani_ml.speech import asr, tts
from sanjeevani_ml.speech.asr import AsrResult
from sanjeevani_ml.speech.bhashini_client import BhashiniError
from sanjeevani_ml.speech.tts import TtsResult

AUDIO = base64.b64encode(b"RIFF----WAVEfmt placeholder bytes").decode()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_speak_returns_audio(client, monkeypatch):
    monkeypatch.setattr(tts, "synthesize",
                        lambda text, language, gender="female": TtsResult(b"\x00wav", language))
    r = client.post("/speech/speak", json={"text": "आपको क्या तकलीफ़ है?", "language": "hi"})
    assert r.status_code == 200, r.text
    assert base64.b64decode(r.json()["audio_base64"]) == b"\x00wav"
    assert r.json()["language"] == "hi"


def test_transcribe_returns_text_and_confidence(client, monkeypatch):
    monkeypatch.setattr(asr, "transcribe",
                        lambda audio, language, audio_format="wav": AsrResult("सीने में दर्द", 0.9, language))
    r = client.post("/speech/transcribe", json={"data_base64": AUDIO, "language": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "सीने में दर्द"
    assert r.json()["confidence"] == 0.9


def test_unknown_confidence_stays_null(client, monkeypatch):
    """Bhashini does not always score a transcription.

    None must reach the caller as None. Substituting 1.0 would present a guess about a
    patient's symptom as certainty, which is the one direction this must never round.
    """
    monkeypatch.setattr(asr, "transcribe",
                        lambda audio, language, audio_format="wav": AsrResult("बुखार", None, language))
    r = client.post("/speech/transcribe", json={"data_base64": AUDIO, "language": "hi"})
    assert r.json()["confidence"] is None


@pytest.mark.parametrize("path,body", [
    ("/speech/speak", {"text": "hello", "language": "hi"}),
    ("/speech/transcribe", {"data_base64": AUDIO, "language": "hi"}),
])
def test_an_outage_is_503_not_500(client, monkeypatch, path, body):
    """The kiosk is expected to fall back to touch and carry on.

    A 500 reads as "this is broken"; a 503 reads as "this capability is away". The
    difference decides whether a patient is turned away from a working interview.
    """
    def boom(*a, **kw):
        raise BhashiniError("BHASHINI_INFERENCE_KEY is not set")

    monkeypatch.setattr(tts, "synthesize", boom)
    monkeypatch.setattr(asr, "transcribe", boom)
    r = client.post(path, json=body)
    assert r.status_code == 503
    assert "touch" in r.json()["detail"] or "on screen" in r.json()["detail"]


def test_an_unsupported_language_is_the_callers_mistake(client, monkeypatch):
    """400, not 503 — nothing is down, the request was wrong."""
    def unsupported(*a, **kw):
        raise ValueError("No known TTS service ID for language 'zz'.")

    monkeypatch.setattr(tts, "synthesize", unsupported)
    r = client.post("/speech/speak", json={"text": "hello", "language": "zz"})
    assert r.status_code == 400


def test_the_transcript_is_never_logged(client, monkeypatch, caplog):
    """It is the patient's own words about their symptoms."""
    secret = "मुझे सीने में तेज़ दर्द है"
    monkeypatch.setattr(asr, "transcribe",
                        lambda audio, language, audio_format="wav": AsrResult(secret, 0.8, language))
    with caplog.at_level("DEBUG"):
        client.post("/speech/transcribe", json={"data_base64": AUDIO, "language": "hi"})
    assert secret not in caplog.text
