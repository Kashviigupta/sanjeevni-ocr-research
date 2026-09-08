"""Sends a detected RedFlag to triage staff.

This service does not call the backend directly per the architecture rule
("this service is called; it does not call") -- so in the real deployment,
the backend is expected to POLL or receive red flags via whatever channel
service/main.py exposes them through, not this module reaching out on its
own. What this module does today: log the flag (no PHI), and -- only if a
triage webhook URL is explicitly configured -- forward it. If that URL isn't
configured, this says so honestly instead of pretending the alert went
anywhere, per the project's "no cheating" rule.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

from sanjeevani_ml.schemas import RedFlag

logger = logging.getLogger("sanjeevani.redflags")


class AlertResult:
    def __init__(self, delivered: bool, detail: str):
        self.delivered = delivered
        self.detail = detail


def fire_alert(flag: RedFlag, session_id: str) -> AlertResult:
    """Log a red flag and forward it if TRIAGE_WEBHOOK_URL is configured.

    Logs ONLY: session_id, reason code, severity, question_id, timestamp.
    Never logs the patient's narration, audio, or free-text answers -- the
    `reason` field is a machine code (e.g. "chest_pain_with_dyspnoea"), not
    patient-supplied text, so it's safe to log per the project's PHI rule.
    """
    logger.warning(
        "RED_FLAG session=%s reason=%s severity=%s question_id=%s",
        session_id, flag.reason, flag.severity, flag.question_id,
    )

    webhook_url = os.environ.get("TRIAGE_WEBHOOK_URL")
    if not webhook_url:
        detail = (
            "TRIAGE_WEBHOOK_URL is not configured -- this alert was logged "
            "locally only and was NOT delivered to any triage system. "
            "This is the honest state until the backend team wires up a "
            "real triage endpoint for this service to call."
        )
        logger.warning(detail)
        return AlertResult(delivered=False, detail=detail)

    try:
        response = requests.post(
            webhook_url,
            json={
                "session_id": session_id,
                "reason": flag.reason,
                "severity": flag.severity,
                "question_id": flag.question_id,
                "fired_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=5,
        )
        response.raise_for_status()
        return AlertResult(delivered=True, detail=f"Delivered, HTTP {response.status_code}")
    except requests.RequestException as exc:
        detail = f"Delivery FAILED: {exc}"
        logger.error(detail)
        return AlertResult(delivered=False, detail=detail)
