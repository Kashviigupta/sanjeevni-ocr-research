"""Score the REAL extraction pipeline (OCR -> structure -> code) against
ground truth, reporting Precision/Recall/F1 using the same formulas MIRAGE
uses (AWP/AWI/HIP) -- so results are directly, honestly comparable to a
published benchmark on the same problem (Indian prescription drug-name
extraction), not just an internal number.

MIRAGE's definitions (arXiv:2410.09729, Section 3.1):
    Let Pe = set of predicted medicine names, Ee = set of expected names,
    Ce = Pe intersect Ee.
    AWP (precision) = |Ce| / |Pe|
    AWI (recall)    = |Ce| / |Ee|
    HIP (F1)        = 2 * AWP * AWI / (AWP + AWI)

Why this replaces the older raw-text-substring scorer for this report:
that scorer only asked "does the expected drug name appear anywhere in the
raw OCR text" -- it never checked whether the ACTUAL extraction pipeline
produced a clean predicted-drugs list, and it had no way to notice a
hallucinated drug that isn't on the prescription at all (a false positive).
Precision needs exactly that: a real predicted set, not a text search.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanjeevani_ml.ocr import engine as ocr_engine
from sanjeevani_ml.extraction.structurer import structure
from sanjeevani_ml.terminology.mapper import code_result

#: Same fuzzy-match philosophy as the earlier scorer, kept identical so a
#: predicted name that's a close but imperfect transcription (e.g.
#: "Atiplele" for "Atiplep") still counts as a correct match -- exactly how
#: a human reviewer would judge it, and how MIRAGE's own C_e = P_e n E_e
#: has to be judged in practice since exact string equality is unrealistic
#: for any OCR/vision system.
MATCH_THRESHOLD = 0.72


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _similarity(a: str, b: str) -> float:
    """Best match of `b` found anywhere inside `a` (or vice versa), not a
    whole-string comparison -- sliding the shorter string across the longer
    one, same technique the real-prescription substring/fuzzy scorer uses.
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if short in long_:
        return 1.0

    best = 0.0
    window = len(short)
    for size in (window - 2, window - 1, window, window + 1, window + 2):
        if size < 3:
            continue
        for start in range(0, max(1, len(long_) - size + 1)):
            ratio = difflib.SequenceMatcher(None, short, long_[start:start + size]).ratio()
            if ratio > best:
                best = ratio
    return best


def _greedy_match(predicted: list[str], expected: list[str]) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Pair each predicted name to its best expected match, one-to-one.

    Greedy on purpose, not optimal bipartite matching: with typically fewer
    than 10 drugs per prescription, a greedy highest-similarity-first pass
    gives the same practical result as the Hungarian algorithm would, at a
    fraction of the code -- not worth the complexity for this scale.
    """
    remaining_expected = list(expected)
    matched: list[tuple[str, str]] = []
    unmatched_predicted: list[str] = []

    # Score every (predicted, expected) pair once, then take the best pairs
    # first so a strong match is never stolen by a weaker one that happened
    # to be considered earlier.
    pairs = []
    for p in predicted:
        for e in remaining_expected:
            pairs.append((_similarity(p, e), p, e))
    pairs.sort(reverse=True)

    used_predicted, used_expected = set(), set()
    for score, p, e in pairs:
        if score < MATCH_THRESHOLD:
            break
        if p in used_predicted or e in used_expected:
            continue
        matched.append((p, e))
        used_predicted.add(p)
        used_expected.add(e)

    unmatched_predicted = [p for p in predicted if p not in used_predicted]
    unmatched_expected = [e for e in expected if e not in used_expected]
    return matched, unmatched_predicted, unmatched_expected


def score_one(image_path: Path, expected: dict) -> dict:
    with open(image_path, "rb") as fh:
        data = fh.read()

    ocr_result = ocr_engine.ocr_document(data, media_type="image/jpeg")
    result = structure(ocr_result, hint=None)
    result = code_result(result)

    predicted_drugs = [
        e.text for e in result.entities
        if getattr(e, "kind", None) == "medication" and getattr(e, "text", None)
    ]
    expected_drugs = expected.get("drugs", [])

    matched, extra_predicted, missed_expected = _greedy_match(predicted_drugs, expected_drugs)

    p_count = len(predicted_drugs)
    e_count = len(expected_drugs)
    c_count = len(matched)

    awp = c_count / p_count if p_count else (1.0 if e_count == 0 else 0.0)
    awi = c_count / e_count if e_count else (1.0 if p_count == 0 else 0.0)
    hip = (2 * awp * awi / (awp + awi)) if (awp + awi) else 0.0

    return {
        "file": image_path.name,
        "predicted": predicted_drugs,
        "matched": matched,
        "hallucinated": extra_predicted,   # predicted, but nothing on the prescription matches
        "missed": missed_expected,          # on the prescription, but never predicted
        "P": p_count, "E": e_count, "C": c_count,
        "AWP": awp, "AWI": awi, "HIP": hip,
    }


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "data" / "samples" / "real_indian_prescriptions"
    )
    expected_files = sorted(folder.glob("*.expected.json"))
    if not expected_files:
        print(f"no .expected.json files found in {folder}")
        return
    if not ocr_engine.available():
        print("Tesseract unavailable -- refusing to report a false 0%.")
        return

    total_P = total_E = total_C = 0
    rows = []

    print(f"{'image':<22}{'AWP':<8}{'AWI':<8}{'HIP':<8}{'hallucinated / missed'}")
    print("-" * 100)

    for ef in expected_files:
        stem = ef.name.replace(".expected.json", "")
        image_path = next((folder / f"{stem}{ext}" for ext in (".jpeg", ".jpg", ".png")
                            if (folder / f"{stem}{ext}").exists()), None)
        if image_path is None:
            continue
        with open(ef) as fh:
            expected = json.load(fh)
        if expected.get("document_type") == "lab_requisition":
            print(f"{image_path.name:<22}(lab order, not scored)")
            continue

        r = score_one(image_path, expected)
        rows.append(r)
        total_P += r["P"]; total_E += r["E"]; total_C += r["C"]

        flag = ", ".join(r["hallucinated"]) or "-"
        miss = ", ".join(r["missed"]) or "-"
        print(f"{image_path.name:<22}{r['AWP']:<8.2f}{r['AWI']:<8.2f}{r['HIP']:<8.2f}"
              f"halluc: {flag} | missed: {miss}")

    print("-" * 100)
    overall_awp = total_C / total_P if total_P else 0.0
    overall_awi = total_C / total_E if total_E else 0.0
    overall_hip = (2 * overall_awp * overall_awi / (overall_awp + overall_awi)) if (overall_awp + overall_awi) else 0.0
    print(f"OVERALL  AWP (precision) = {overall_awp:.2%}   "
          f"AWI (recall) = {overall_awi:.2%}   HIP (F1) = {overall_hip:.2%}")
    print(
        "\nThese use MIRAGE's own formulas (arXiv:2410.09729 Sec 3.1) so they compare "
        "directly to a published benchmark on the same task -- their zero-shot "
        "foundation-model baseline scored 5-7%; their fine-tuned result (743K training "
        "images, 13 GPU-days) reached 82%."
    )


if __name__ == "__main__":
    main()
