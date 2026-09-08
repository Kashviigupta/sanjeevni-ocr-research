"""Environment-driven settings.

Every capability here is optional and degrades: no Tesseract binary and no API key
still leaves a service that starts, answers /health honestly, and lets the other half
of the team keep working.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]   # ml/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"

    # --- OCR -----------------------------------------------------------------
    #: Absolute path to the tesseract binary. Needed on Windows, where it is not on PATH
    #: by default: C:\Program Files\Tesseract-OCR\tesseract.exe
    tesseract_cmd: str | None = None
    #: Default OCR languages. "eng+hin" once tesseract-ocr-hin is installed.
    ocr_languages: str = "eng"
    #: Below this, we tell the patient the scan was poor rather than guessing.
    min_word_confidence: float = 0.35
    #: PDF pages are rasterised at this DPI. 200 is the sweet spot: 300 is ~2x slower
    #: for marginal accuracy gain on printed lab reports.
    pdf_dpi: int = 200
    max_pages: int = 10

    # --- Optional vision model ----------------------------------------------
    #: Handwriting is where Tesseract falls apart and a vision model earns its keep.
    #: Absent key = the offline path, which must always work.
    anthropic_api_key: str | None = None
    vision_model: str = "claude-opus-5"
    #: Alternative vision provider. Checked if anthropic_api_key is absent — the
    #: repo has been run with a Gemini key in practice, so this is a real,
    #: exercised path, not a hypothetical.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    #: Only escalate to the vision model when OCR confidence is this poor.
    vision_fallback_threshold: float = 0.55

    # --- Terminology ---------------------------------------------------------
    data_dir: Path = ROOT / "data"
    #: Fuzzy-match cutoff for mapping free text to a curated term. Above this we accept
    #: the code; below it we return the raw text with code=None. Deliberately strict —
    #: a wrong LOINC code is worse than no code.
    term_match_threshold: float = 0.86

    @property
    def vision_enabled(self) -> bool:
        return bool(self.anthropic_api_key or self.gemini_api_key)

    @property
    def vision_provider(self) -> str | None:
        """Which vision backend is actually configured, or None if neither key is set.

        Anthropic wins if both happen to be set — arbitrary but needs a rule, and
        Anthropic was the originally designed provider per the code comments in
        ocr/engine.py.
        """
        if self.anthropic_api_key:
            return "anthropic"
        if self.gemini_api_key:
            return "gemini"
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# pytesseract reads this global rather than an argument, so set it once at import.
if settings.tesseract_cmd:
    os.environ.setdefault("TESSERACT_CMD", settings.tesseract_cmd)
