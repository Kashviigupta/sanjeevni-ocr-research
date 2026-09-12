"""Context-aware correction: take a raw, possibly-misspelled OCR/vision
transcription of a drug name, and use an LLM (text-only, cheap, no image
re-send) to pick the correct spelling from a SHORTLIST of real, known drug
brands -- rather than trusting whatever the vision model free-generated.

Why this exists: the strict-vs-fuzzy gap measured on real prescriptions
(64% vs 82%) shows the vision model usually recognises the RIGHT drug but
doesn't always spell it correctly ("Atiplele" for "Atiplep"). That's a
constrained-choice problem, not a free-generation problem -- and LLMs are
much more reliable at "which of these five real names is this" than at
"transcribe this squiggle from scratch." This targets exactly that gap.

Two-step design, deliberately kept separate from the vision OCR call:
  1. `shortlist_candidates()` -- fast, local, no API call. Uses the same
     difflib approach as the scorer to find the N most similar known brand
     names from drugs_seed.csv. Free and instant.
  2. `correct_with_llm()` -- one cheap text-only Gemini call per uncertain
     drug name, given the raw text AND the shortlist, asked to either pick
     one or say none fit. Never asked to invent a name outside the list --
     this is the same "never invent a code" discipline the terminology
     mapper already follows, applied one layer earlier in the pipeline.
"""
from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanjeevani_ml.config import settings

#: How many candidates to show the LLM. Small on purpose -- a long list
#: dilutes the model's attention and makes "none of these fit" harder to
#: judge confidently; the shortlist should already contain the answer if
#: one exists.
SHORTLIST_SIZE = 5


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_all_known_brands(drugs_csv: Path) -> list[str]:
    """Every brand name across every row of the terminology table, flattened.

    Reads the file directly rather than going through the mapper, because
    we need the full brand list to search against -- the mapper is built
    for exact/near lookup of one name, not "give me everything."
    """
    brands: list[str] = []
    with open(drugs_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for brand in (row.get("brands") or "").split("|"):
                brand = brand.strip()
                if brand:
                    brands.append(brand)
    return brands


def shortlist_candidates(raw_name: str, known_brands: list[str], n: int = SHORTLIST_SIZE) -> list[str]:
    """The N known brand names most textually similar to `raw_name`.

    Pure difflib, no API call -- this step is free and instant, and it's
    what actually constrains the LLM call that follows: the model is never
    asked to invent a name, only to choose among real, verified entries
    already in our own terminology table.
    """
    scored = sorted(
        known_brands,
        key=lambda b: difflib.SequenceMatcher(None, _normalize(raw_name), _normalize(b)).ratio(),
        reverse=True,
    )
    # De-duplicate while preserving order (a brand can repeat if it's an
    # alias on more than one row, which does happen in this table).
    seen, unique = set(), []
    for b in scored:
        if b not in seen:
            unique.append(b)
            seen.add(b)
        if len(unique) >= n:
            break
    return unique


def correct_with_llm(raw_name: str, shortlist: list[str], line_context: str = "") -> str | None:
    """Ask Gemini (text-only) to pick the correct spelling from `shortlist`,
    or return None if it judges none of them are a plausible match.

    Text-only and cheap: this does NOT re-send the image. It only reasons
    over strings, which is a much smaller, faster, cheaper call than a
    vision request -- and it's the reason this step is worth adding even
    given the same daily API quota constraints already documented for the
    vision fallback.
    """
    if not settings.allow_hosted_vision:
        # Same safety gate as the vision fallback -- a text correction call
        # is still a hosted API call, and the same opt-in discipline applies.
        return None

    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)

    prompt = (
        "A handwriting-recognition system transcribed a medicine name from a "
        f"real Indian doctor's prescription as: {raw_name!r}\n\n"
        f"The surrounding line of text was: {line_context!r}\n\n"
        "Here are the ONLY real, verified medicine brand names it could "
        f"plausibly be, based on textual similarity: {shortlist}\n\n"
        "Reply with EXACTLY ONE of these:\n"
        "- One of the candidate names above, copied EXACTLY as written in the list, "
        "if you are confident that is the intended drug.\n"
        "- The single word NONE if you are not confident any of them is correct.\n\n"
        "Do not explain your answer. Do not suggest a name that is not in the list. "
        "Do not guess if you are not reasonably confident."
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[prompt],
    )
    answer = (response.text or "").strip()

    if answer.upper() == "NONE":
        return None
    # Only accept the answer if it's an exact member of the shortlist we
    # gave it -- never trust free-form output here, even if it looks close.
    # This is the same "never invent" discipline as the terminology mapper.
    for candidate in shortlist:
        if candidate.strip().lower() == answer.strip().lower():
            return candidate
    return None


def correct_prediction_list(raw_names: list[str], drugs_csv: Path,
                             line_contexts: dict[str, str] | None = None) -> dict[str, str | None]:
    """Run the full two-step correction over a list of raw predicted names.

    Returns {raw_name: corrected_name_or_None}. A None means "checked, and
    genuinely nothing in the terminology table looks like a confident
    match" -- which is itself useful information (this drug may be a
    combination product not yet in the table, per the earlier findings in
    this project, rather than a spelling problem at all).
    """
    known_brands = _load_all_known_brands(drugs_csv)
    line_contexts = line_contexts or {}

    results: dict[str, str | None] = {}
    for raw in raw_names:
        shortlist = shortlist_candidates(raw, known_brands)
        corrected = correct_with_llm(raw, shortlist, line_contexts.get(raw, ""))
        results[raw] = corrected
    return results


if __name__ == "__main__":
    # Quick manual smoke test against the known real-world near-misses
    # already documented for this project, so you can see it working
    # before wiring it into the full scorer.
    drugs_csv = Path(__file__).resolve().parents[1] / "data" / "drugs_seed.csv"
    test_cases = ["Atiplele", "Devoxim", "Super", "Crucial MF"]

    known_brands = _load_all_known_brands(drugs_csv)
    for raw in test_cases:
        shortlist = shortlist_candidates(raw, known_brands)
        print(f"\nraw: {raw!r}")
        print(f"  shortlist: {shortlist}")
        corrected = correct_with_llm(raw, shortlist)
        print(f"  LLM correction: {corrected!r}")
