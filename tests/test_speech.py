"""Tests for speech/asr.py and speech/tts.py.

These mock Bhashini's HTTP response rather than hitting the real network --
that keeps tests fast, free, and runnable with no API key present, while
proving our request-building and response-parsing logic is correct.

This does NOT replace a real-audio test. ml/scripts/test_bhashini.py already
proved the live connection + Hindi ASR service ID work against our real key.
A real Hindi speech fixture test (not silence, not mocked) is still needed
before this module is "done" per the project brief.
"""

from unittest.mock import patch

import pytest

from sanjeevani_ml.speech.asr import transcribe
from sanjeevani_ml.speech.bhashini_client import BhashiniError
from sanjeevani_ml.speech.tts import synthesize


def test_transcribe_parses_bhashini_response():
    fake_response = {
        "pipelineResponse": [
            {"taskType": "asr", "output": [{"source": "मुझे बुखार है", "confidence": 0.91}]}
        ]
    }
    with patch("sanjeevani_ml.speech.asr.compute", return_value=fake_response):
        result = transcribe(audio_bytes=b"fake-wav-bytes", language="hi")

    assert result.text == "मुझे बुखार है"
    assert result.confidence == 0.91
    assert result.language == "hi"


def test_transcribe_rejects_unsupported_language():
    with pytest.raises(ValueError):
        transcribe(audio_bytes=b"irrelevant", language="ta")


def test_transcribe_raises_on_bad_response_shape():
    with patch("sanjeevani_ml.speech.asr.compute", return_value={"unexpected": "shape"}):
        with pytest.raises(BhashiniError):
            transcribe(audio_bytes=b"fake-wav-bytes", language="hi")


def test_synthesize_parses_bhashini_response():
    fake_response = {
        "pipelineResponse": [
            {"taskType": "tts", "audio": [{"audioContent": "ZmFrZS1hdWRpby1ieXRlcw=="}]}
        ]
    }
    with patch("sanjeevani_ml.speech.tts.compute", return_value=fake_response):
        result = synthesize(text="क्या आपको बुखार है?", language="hi")

    assert result.audio_bytes == b"fake-audio-bytes"
    assert result.language == "hi"


def test_synthesize_rejects_unsupported_language():
    with pytest.raises(ValueError):
        synthesize(text="irrelevant", language="ta")
