"""Low-level HTTP client for Bhashini's Dhruva inference endpoint.

Bhashini's "proper" flow is two calls (Pipeline Config, then Pipeline
Compute). For a known, fixed set of models (see the service ID dicts below)
we skip the Config call and go straight to Compute -- confirmed to work via
ml/scripts/test_bhashini.py (the smoke-test tool), which proved:
  - the endpoint below is correct
  - the Authorization header is just the raw inference key (no name/value pair)
  - "hi" ASR with serviceId "ai4bharat/conformer-hi-gpu--t4" is ACCEPTED

Anything not marked "confirmed" below is still a docs-sourced guess and
should be smoke-tested the same way before being relied on.
"""

from __future__ import annotations

import os

import requests

COMPUTE_ENDPOINT = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

# ASR service IDs, by source language.
ASR_SERVICE_IDS: dict[str, str] = {
    "hi": "ai4bharat/conformer-hi-gpu--t4",  # CONFIRMED via test_bhashini.py
    "en": "ai4bharat/whisper-medium-en--gpu--t4",  # CONFIRMED via test_bhashini.py
}

# TTS service IDs, by source language. Neither of these has been smoke-tested
# against our key yet -- both are still docs-sourced guesses.
TTS_SERVICE_IDS: dict[str, str] = {
    "hi": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",  # UNCONFIRMED
    "en": "ai4bharat/indic-tts-coqui-misc-gpu--t4",  # UNCONFIRMED
}


class BhashiniError(RuntimeError):
    """Raised when Bhashini's API returns something other than 200 OK,
    or when its response isn't shaped the way our code expects."""


def _inference_key() -> str:
    key = os.environ.get("BHASHINI_INFERENCE_KEY")
    if not key:
        raise BhashiniError(
            "BHASHINI_INFERENCE_KEY is not set. Add it to ml/.env and make "
            "sure something calls load_dotenv() before this runs."
        )
    return key


def compute(payload: dict) -> dict:
    """POST a pipelineTasks/inputData payload to Bhashini and return the JSON response.

    Raises BhashiniError on any non-200 response, with the response body
    included, so a caller never silently gets an empty/garbage result.
    """
    response = requests.post(
        COMPUTE_ENDPOINT,
        headers={
            "Authorization": _inference_key(),
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    if response.status_code != 200:
        raise BhashiniError(
            f"Bhashini returned {response.status_code}: {response.text}"
        )
    return response.json()
