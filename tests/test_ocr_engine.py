"""Which vision provider gets to see a patient's document, and when.

These exist because the Anthropic-to-Gemini switch left `settings.vision_provider`,
`gemini_api_key` and `gemini_model` referenced in `ocr/engine.py` but never defined in
`config.py`. Every access raised AttributeError inside a bare `except Exception`, so
handwriting escalation silently never fired and nothing anywhere reported a problem.

A silent permanent degradation is worse than a crash: the demo shows a feature that
was never running.
"""
import pytest

from sanjeevani_ml.config import Settings


def test_every_attribute_the_vision_path_reads_actually_exists():
    """The bug was a missing field, not bad logic. Assert the fields exist."""
    s = Settings()
    for field in ("vision_provider", "vision_enabled", "gemini_api_key",
                  "gemini_model", "anthropic_api_key", "anthropic_model",
                  "allow_hosted_vision", "vision_base_url"):
        getattr(s, field)


def test_a_hosted_key_alone_does_not_start_sending_documents_out():
    """Opting in must be deliberate. A key in someone's .env is not consent."""
    assert Settings(gemini_api_key="k").vision_provider is None
    assert Settings(anthropic_api_key="k").vision_provider is None


def test_opting_in_selects_the_hosted_provider():
    assert Settings(gemini_api_key="k", allow_hosted_vision=True).vision_provider == "gemini"


def test_self_hosted_wins_and_needs_no_opt_in():
    """MedGemma is the only provider allowed to see a real patient's document."""
    s = Settings(vision_base_url="http://medgemma:8080", gemini_api_key="k",
                 allow_hosted_vision=True)
    assert s.vision_provider == "self_hosted"


def test_prod_refuses_to_start_with_a_hosted_provider():
    """Free tiers train on submissions. In prod that is an unconsented disclosure."""
    with pytest.raises(ValueError, match="ALLOW_HOSTED_VISION"):
        Settings(env="prod", allow_hosted_vision=True, gemini_api_key="k")


def test_self_hosted_configured_but_unwired_falls_back_offline_not_outward():
    """MedGemma is not implemented yet. That must degrade to Tesseract, never
    quietly reroute the document to a third party because the good option is absent."""
    from sanjeevani_ml.ocr import engine

    assert engine.run_vision_ocr(b"x", "image/jpeg") is None


# --- language codes ------------------------------------------------------------
# The platform speaks ISO 639-1 end to end: a session's language is "hi" or "en" and
# the backend hands that value straight to the OCR engine. Tesseract speaks ISO 639-2/T
# and raises on a two-letter code. The raise was caught, logged, and returned as an
# empty page with 0.00 confidence — so every document upload read nothing, in every
# language the product actually ships, and reported success.

#: Pin what is installed, so these test the MAPPING rather than whatever language
#: packs happen to be on the machine running them. Without this they pass on a laptop
#: with no Tesseract and fail on one that has English only - which is exactly what
#: happened, and the code was right both times.
@pytest.fixture()
def all_packs_installed(monkeypatch):
    from sanjeevani_ml.ocr import engine

    monkeypatch.setattr(
        engine, "installed_languages",
        lambda: frozenset({"eng", "hin", "mar", "ben", "tam", "tel"}),
    )


def test_app_locales_become_tesseract_languages(all_packs_installed):
    from sanjeevani_ml.ocr.engine import resolve_language

    assert resolve_language("en") == "eng"
    assert resolve_language("hi") == "hin"
    assert resolve_language("mr") == "mar"


def test_tesseract_strings_pass_through_untouched(all_packs_installed):
    """Callers that already know what they are doing must keep working."""
    from sanjeevani_ml.ocr.engine import resolve_language

    assert resolve_language("eng") == "eng"
    assert resolve_language("eng+hin") == "eng+hin"


def test_a_mixed_locale_maps_every_part(all_packs_installed):
    from sanjeevani_ml.ocr.engine import resolve_language

    assert resolve_language("hi+en") == "hin+eng"


def test_an_unknown_code_falls_back_rather_than_failing():
    """Reading a document in the wrong language still gives a physician something to
    correct. Reading nothing gives them nothing."""
    from sanjeevani_ml.config import settings
    from sanjeevani_ml.ocr.engine import resolve_language

    assert resolve_language("zz") == settings.ocr_languages
    assert resolve_language("") == settings.ocr_languages
    assert resolve_language(None) == settings.ocr_languages


def test_the_language_the_backend_actually_sends_is_supported():
    """The intake router passes `session.language` verbatim. Those are the values."""
    from sanjeevani_ml.ocr.engine import resolve_language

    for locale in ("hi", "en"):
        resolved = resolve_language(locale)
        assert len(resolved) >= 3, f"{locale} resolved to {resolved!r}, which Tesseract rejects"


def test_a_language_pack_that_is_not_installed_is_dropped(monkeypatch):
    """`tesseract-ocr-hin` is a separate apt package.

    Asking for a pack that is not present raises the same TesseractError as a malformed
    code, and produces the same silent empty page - so defaulting to "eng+hin" on a
    machine with only English would just move the bug rather than fix it.
    """
    from sanjeevani_ml.ocr import engine

    monkeypatch.setattr(engine, "installed_languages", lambda: frozenset({"eng"}))
    assert engine.resolve_language("eng+hin") == "eng"
    assert engine.resolve_language("hi") == "eng"


def test_an_installed_pack_is_kept(monkeypatch):
    from sanjeevani_ml.ocr import engine

    monkeypatch.setattr(engine, "installed_languages", lambda: frozenset({"eng", "hin"}))
    assert engine.resolve_language("eng+hin") == "eng+hin"
    assert engine.resolve_language("hi") == "hin"


def test_resolution_never_returns_an_empty_string(monkeypatch):
    """An empty lang string is passed straight to Tesseract and fails there."""
    from sanjeevani_ml.ocr import engine

    for installed in (frozenset(), frozenset({"eng"}), frozenset({"hin"})):
        monkeypatch.setattr(engine, "installed_languages", lambda i=installed: i)
        for requested in (None, "", "zz", "hi", "eng+hin"):
            assert engine.resolve_language(requested), f"empty for {requested!r} with {installed}"
