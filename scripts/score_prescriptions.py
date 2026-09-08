"""Score the extraction pipeline against the synthetic prescription set.

Every image from `make_prescriptions.py` ships with a `.expected.json` naming the drugs
that are actually on it, so accuracy is a number rather than an impression. Run this
before and after a change to handwriting handling and you can say whether it helped.

Three things are measured separately, because they fail for different reasons:

* **OCR recall** — did the characters come back at all? Fails on handwriting, blur,
  low light.
* **Brand recognition** — did "Zerodol" survive as a token? Fails on OCR noise.
* **Coding** — did the brand resolve to an RxNorm ingredient? Fails when the
  terminology table has a gap, which is a data problem, not a vision one.

A drug can be read perfectly and still not code, and the fix is different in each case.

Usage:
    python scripts/score_prescriptions.py
    python scripts/score_prescriptions.py --dir data/samples/prescriptions --verbose

Needs Tesseract on PATH (or `TESSERACT_CMD` set). Without it the OCR engine degrades to
`engine="unavailable"` and every score is zero — that is the engine behaving correctly,
not the images being hard, and this script says so rather than reporting a false 0%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sanjeevani_ml.ocr.engine import ocr_document
from sanjeevani_ml.terminology.mapper import _drug_terminology_code


def _coded(brand: str) -> str | None:
    """The RxNorm ingredient a brand resolves to, or None if the table has no entry.

    This is the whole point of the brand column: a script says "Zerodol", and only the
    terminology table knows that is aceclofenac, RxCUI 16689.
    """
    code = _drug_terminology_code(brand)
    return getattr(code, "code", None) if code else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/samples/prescriptions"))
    ap.add_argument("--verbose", action="store_true", help="list every miss")
    args = ap.parse_args()

    images = sorted(args.dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {args.dir} - run scripts/make_prescriptions.py first")

    by_condition: dict[str, list[float]] = {}
    total_found = total_expected = total_codeable = 0
    unavailable = 0

    print(f"{'image':<32} {'conf':>5} {'read':>7} {'coded':>7}  misses")
    print("-" * 92)

    for image in images:
        meta = json.loads(image.with_suffix("").with_suffix(".expected.json")
                          .read_text(encoding="utf-8"))
        expected = [d["brand"] for d in meta["drugs"]]

        result = ocr_document(image.read_bytes(), media_type="image/jpeg")
        if result.engine == "unavailable":
            unavailable += 1
            print(f"{image.name:<32}     -       -       -  OCR engine unavailable")
            continue

        text = result.text.lower()
        found = [b for b in expected if b.lower() in text]
        coded = [b for b in found if _coded(b)]
        missed = [b for b in expected if b not in found]

        total_found += len(found)
        total_expected += len(expected)
        total_codeable += len(coded)
        by_condition.setdefault(meta["condition"], []).append(len(found) / len(expected))

        print(f"{image.name:<32} {result.mean_confidence:>5.2f} "
              f"{len(found):>3}/{len(expected):<3} {len(coded):>3}/{len(found) or 0:<3}  "
              f"{', '.join(missed) if missed else '-'}")
        if args.verbose and missed:
            print(f"    OCR text: {result.text[:160]!r}")

    print("-" * 92)
    if unavailable == len(images):
        print("Tesseract is not installed or not on PATH, so nothing was measured.")
        print("Install it and set TESSERACT_CMD, then re-run. The zeros above are the")
        print("engine degrading correctly - not a pipeline result.")
        return

    print(f"brand recall : {total_found}/{total_expected} "
          f"({100 * total_found / max(total_expected, 1):.0f}%)")
    print(f"coded        : {total_codeable}/{max(total_found, 1)} of what was read "
          f"({100 * total_codeable / max(total_found, 1):.0f}%)")
    print()
    print("by condition:")
    for condition, scores in sorted(by_condition.items(),
                                    key=lambda kv: -sum(kv[1]) / len(kv[1])):
        mean = 100 * sum(scores) / len(scores)
        bar = "#" * int(mean / 5)
        print(f"  {condition:<14} {mean:>5.0f}%  {bar}")

    print()
    print("Read the two numbers separately: low recall is a vision problem, low coding")
    print("with high recall is a terminology gap - a missing row, not a missing model.")


if __name__ == "__main__":
    main()
