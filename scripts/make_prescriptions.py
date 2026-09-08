"""Generate synthetic prescription images for OCR and extraction testing.

**Every patient, doctor, clinic and registration number here is invented.** That is the
point. Real prescriptions cannot be used for this: the hosted vision fallback runs on a
free tier that trains on what it is sent, so a real patient's paperwork going through it
is an unconsented disclosure under the DPDP Act — see docs/THREAT-MODEL.md T-3. These
images exercise the pipeline identically and carry none of that.

What makes them useful rather than decorative is that they are *hard* in the ways real
OPD paperwork is hard:

* **Handwriting.** Four different fonts, so the pipeline is not tuned to one hand.
* **Indian brand names.** A script says "Zerodol", not "aceclofenac". Resolving the
  brand is the actual work, and every brand used here is one the terminology table
  can currently code.
* **Photographed, not scanned.** Skew, fold shadows, uneven phone-flash lighting,
  sensor noise and low contrast — because a patient hands over a creased paper and a
  kiosk camera photographs it at an angle.
* **Legible degradation.** A human can still read every one of these. If OCR cannot,
  that is a pipeline finding, not an unfair test.

Usage:
    python scripts/make_prescriptions.py --out data/samples/prescriptions
    python scripts/make_prescriptions.py --out /tmp/rx --count 12 --seed 7

Each image is written alongside a `.expected.json` naming the drugs it contains, so
extraction accuracy can be scored rather than eyeballed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# --------------------------------------------------------------------- fixtures
# Invented. Any resemblance to a real clinic or practitioner is accidental, and the
# registration numbers are deliberately malformed so they cannot match a real one.
CLINICS = [
    ("Sunrise Polyclinic", "Plot 14, Sector 8, Rohini, Delhi", "REG/FAKE/0001"),
    ("Anand Wellness Centre", "22 MG Road, Indiranagar, Bengaluru", "REG/FAKE/0002"),
    ("Shanti General Clinic", "Near Bus Stand, Kalyan West, Thane", "REG/FAKE/0003"),
    ("Nirmal Health Point", "Ward 6, Civil Lines, Nagpur", "REG/FAKE/0004"),
]
DOCTORS = [
    ("Dr. R. Venkatesan", "MBBS, MD (Gen Med)"),
    ("Dr. Priya Nair", "MBBS, DNB"),
    ("Dr. A. K. Bhattacharya", "MBBS, MD"),
    ("Dr. S. Deshmukh", "MBBS, DGO"),
]
PATIENTS = [
    ("Ramesh Kumar", 54, "M"), ("Lakshmi Bai", 47, "F"), ("Imran Sheikh", 33, "M"),
    ("Anita Deshpande", 61, "F"), ("Joseph Mathew", 28, "M"), ("Sunita Yadav", 39, "F"),
]

#: (form, brand, generic, strength, sig, duration). Brands only — resolving them is
#: terminology's job, not extraction's. The dosage-form prefix ("Tab.", "Cap.") is how
#: an Indian prescription line is usually written and how the extractor recognises one;
#: a few lines deliberately omit it, because plenty of real scripts do too and the
#: pipeline should not fall silent on them.
DRUG_LINES = [
    ("Tab", "Glycomet", "metformin", "500mg", "1-0-1", "30 days"),
    ("Tab", "Amlong", "amlodipine", "5mg", "0-0-1", "30 days"),
    ("Tab", "Telma", "telmisartan", "40mg", "1-0-0", "30 days"),
    ("Tab", "Atorva", "atorvastatin", "10mg", "0-0-1", "30 days"),
    ("Tab", "Ecosprin", "aspirin", "75mg", "0-1-0", "30 days"),
    ("Tab", "Crocin", "paracetamol", "650mg", "1-1-1", "5 days"),
    ("Tab", "Zerodol", "aceclofenac", "100mg", "1-0-1", "5 days"),
    ("Tab", "Mox", "amoxicillin", "500mg", "1-1-1", "7 days"),
    ("Tab", "Azithral", "azithromycin", "500mg", "1-0-0", "3 days"),
    ("Tab", "Flagyl", "metronidazole", "400mg", "1-1-1", "5 days"),
    ("Cap", "Doxy", "doxycycline", "100mg", "1-0-1", "7 days"),
    ("Tab", "Pantocid", "pantoprazole", "40mg", "1-0-0", "14 days"),
    ("Cap", "Omez", "omeprazole", "20mg", "1-0-0", "14 days"),
    ("Tab", "Domstal", "domperidone", "10mg", "1-1-1", "3 days"),
    ("Tab", "Emeset", "ondansetron", "4mg", "SOS", "3 days"),
    ("Tab", "Levocet", "levocetirizine", "5mg", "0-0-1", "7 days"),
    ("Tab", "Montair", "montelukast", "10mg", "0-0-1", "30 days"),
    ("Syp", "Asthalin", "salbutamol", "2mg", "1-1-1", "7 days"),
    ("Tab", "Wysolone", "prednisolone", "10mg", "1-0-0", "5 days"),
    ("Tab", "Decdan", "dexamethasone", "0.5mg", "1-0-1", "3 days"),
    ("Tab", "Thyronorm", "levothyroxine", "50mcg", "1-0-0", "30 days"),
    ("Tab", "Clopilet", "clopidogrel", "75mg", "0-1-0", "30 days"),
]
ADVICE = [
    "Take after food. Plenty of fluids.",
    "Review after 1 week. BP chart to be maintained.",
    "Avoid oily food. Walk 30 min daily.",
    "Fasting sugar on next visit.",
    "Complete the full course.",
]

FONT_DIR = Path("C:/Windows/Fonts")
#: Four hands, so nothing is tuned to one. Falls back to the default bitmap font on
#: a machine without these, which still produces a usable (if tidier) image.
HANDWRITING = ["Inkfree.ttf", "LHANDW.TTF", "comic.ttf", "FRSCRIPT.TTF"]
PRINTED = ["arial.ttf", "calibri.ttf", "times.ttf"]

PAPER = (252, 250, 242)
INK = [(28, 42, 96), (18, 22, 30), (30, 60, 40)]  # blue, black, green pens


def _font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in names:
        path = FONT_DIR / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _paper(w: int, h: int, rng: random.Random) -> Image.Image:
    """Off-white stock with faint fibre noise — flat white reads as synthetic."""
    img = Image.new("RGB", (w, h), PAPER)
    px = img.load()
    for _ in range(int(w * h * 0.02)):
        x, y = rng.randrange(w), rng.randrange(h)
        d = rng.randint(-8, 4)
        r, g, b = px[x, y]
        px[x, y] = (max(0, r + d), max(0, g + d), max(0, b + d))
    return img


def _draw_prescription(rng: random.Random, handwritten: bool) -> tuple[Image.Image, dict]:
    W, H = 1000, 1400
    img = _paper(W, H, rng)
    d = ImageDraw.Draw(img)

    clinic, address, reg = rng.choice(CLINICS)
    doctor, quals = rng.choice(DOCTORS)
    patient, age, sex = rng.choice(PATIENTS)
    ink = rng.choice(INK)

    head_f = _font(PRINTED, 40)
    sub_f = _font(PRINTED, 22)
    body_f = _font(HANDWRITING if handwritten else PRINTED, 34 if handwritten else 28)
    small_f = _font(HANDWRITING if handwritten else PRINTED, 27 if handwritten else 23)

    # --- letterhead (always printed; clinics have printed pads) ---
    d.text((60, 50), clinic, font=head_f, fill=(20, 40, 90))
    d.text((60, 100), address, font=sub_f, fill=(70, 70, 70))
    d.text((60, 128), f"{doctor}  |  {quals}  |  {reg}", font=sub_f, fill=(70, 70, 70))
    d.line([(60, 170), (W - 60, 170)], fill=(20, 40, 90), width=3)

    # --- patient block (handwritten on a printed pad, as in life) ---
    y = 210
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026"
    d.text((60, y), "Name:", font=sub_f, fill=(90, 90, 90))
    d.text((150, y - 6), patient, font=small_f, fill=ink)
    d.text((600, y), "Date:", font=sub_f, fill=(90, 90, 90))
    d.text((680, y - 6), date, font=small_f, fill=ink)
    y += 46
    d.text((60, y), "Age/Sex:", font=sub_f, fill=(90, 90, 90))
    d.text((175, y - 6), f"{age} / {sex}", font=small_f, fill=ink)
    y += 70

    d.text((60, y), "Rx", font=_font(PRINTED, 52), fill=ink)
    y += 80

    # --- the drugs ---
    chosen = rng.sample(DRUG_LINES, rng.randint(3, 5))
    expected = []
    for i, (form, brand, generic, strength, sig, duration) in enumerate(chosen, 1):
        jitter = rng.randint(-3, 3) if handwritten else 0
        # One line in five drops the dosage form, as real scripts do.
        prefix = "" if rng.random() < 0.2 else f"{form}. "
        d.text((90, y + jitter), f"{i}.", font=body_f, fill=ink)
        d.text((140, y + jitter), f"{prefix}{brand} {strength}", font=body_f, fill=ink)
        d.text((560, y + jitter + 4), f"{sig}   x {duration}", font=small_f, fill=ink)
        expected.append({
            "brand": brand, "generic": generic, "strength": strength,
            "sig": sig, "duration": duration, "form": prefix.strip(" .") or None,
        })
        y += 76

    y += 30
    d.text((60, y), "Advice:", font=sub_f, fill=(90, 90, 90))
    d.text((150, y - 6), rng.choice(ADVICE), font=small_f, fill=ink)

    # --- signature ---
    sig_f = _font(["FRSCRIPT.TTF", "Inkfree.ttf"], 44)
    d.text((W - 380, H - 220), doctor.replace("Dr. ", ""), font=sig_f, fill=ink)
    d.line([(W - 390, H - 160), (W - 90, H - 160)], fill=(120, 120, 120), width=2)
    d.text((W - 340, H - 150), "Signature", font=sub_f, fill=(120, 120, 120))

    meta = {
        "clinic": clinic, "doctor": doctor, "patient": patient,
        "age": age, "sex": sex, "date": date,
        "handwritten": handwritten, "drugs": expected,
        "synthetic": True,
        "note": "Fully invented. Safe for hosted vision models; contains no real patient data.",
    }
    return img, meta


# ------------------------------------------------------------------ degradation
# A kiosk photographs a creased paper under a ceiling tube light at an angle. Each of
# these reproduces one way that goes wrong. All remain human-legible on purpose.

def _skew(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.rotate(rng.uniform(-4.5, 4.5), resample=Image.BICUBIC,
                      expand=True, fillcolor=PAPER)


def _fold(img: Image.Image, rng: random.Random) -> Image.Image:
    """A horizontal fold shadow — paper carried folded in a pocket."""
    w, h = img.size
    overlay = Image.new("L", (w, h), 0)
    od = ImageDraw.Draw(overlay)
    for fy in (int(h * rng.uniform(0.3, 0.38)), int(h * rng.uniform(0.62, 0.7))):
        for off in range(-6, 7):
            od.line([(0, fy + off), (w, fy + off)], fill=max(0, 46 - abs(off) * 7), width=1)
    overlay = overlay.filter(ImageFilter.GaussianBlur(3))
    return Image.composite(Image.new("RGB", (w, h), (140, 138, 130)), img, overlay)


def _flash(img: Image.Image, rng: random.Random) -> Image.Image:
    """Uneven phone-flash falloff: bright hot spot, dark corners."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.25, 0.5) * h
    steps = 60
    for i in range(steps, 0, -1):
        r = (i / steps) * max(w, h) * 0.85
        md.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8],
                   fill=int(110 * (1 - i / steps)))
    mask = mask.filter(ImageFilter.GaussianBlur(60))
    dark = ImageEnhance.Brightness(img).enhance(0.62)
    return Image.composite(img, dark, mask)


def _noise(img: Image.Image, rng: random.Random, amount: int = 12) -> Image.Image:
    px = img.load()
    w, h = img.size
    for _ in range(int(w * h * 0.06)):
        x, y = rng.randrange(w), rng.randrange(h)
        d = rng.randint(-amount, amount)
        r, g, b = px[x, y]
        px[x, y] = (min(255, max(0, r + d)), min(255, max(0, g + d)), min(255, max(0, b + d)))
    return img


def _perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    """Photographed off-axis rather than laid flat on a scanner."""
    w, h = img.size
    dx, dy = w * rng.uniform(0.02, 0.055), h * rng.uniform(0.01, 0.03)
    # Solve the inverse map for the four-corner transform.
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(dx, dy), (w - dx * 0.4, 0), (w, h - dy), (dx * 0.5, h)]
    matrix = []
    for (X, Y), (x, y) in zip(dst, src):
        matrix.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        matrix.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    import numpy as np

    A = np.array(matrix, dtype=float)
    B = np.array([c for pt in src for c in pt], dtype=float)
    coeffs = np.linalg.solve(A, B)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         Image.BICUBIC, fillcolor=PAPER)


CONDITIONS = {
    "clean":       lambda im, rng: im,
    "skewed":      lambda im, rng: _noise(_skew(im, rng), rng),
    "folded":      lambda im, rng: _noise(_fold(_skew(im, rng), rng), rng),
    "phone_flash": lambda im, rng: _noise(_flash(_skew(im, rng), rng), rng, 16),
    "off_axis":    lambda im, rng: _noise(_flash(_perspective(im, rng), rng), rng, 14),
    "low_light":   lambda im, rng: _noise(
        ImageEnhance.Contrast(ImageEnhance.Brightness(_skew(im, rng)).enhance(0.55)).enhance(0.8),
        rng, 18),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/samples/prescriptions", type=Path)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    names = list(CONDITIONS)
    written = 0
    for i in range(args.count):
        condition = names[i % len(names)]
        # Bias toward handwriting: that is the failure mode Tesseract has, and the
        # reason the vision fallback exists at all.
        handwritten = (i % 4) != 3
        img, meta = _draw_prescription(rng, handwritten)
        img = CONDITIONS[condition](img, rng)

        kind = "hand" if handwritten else "print"
        stem = f"rx_{i + 1:02d}_{kind}_{condition}"
        path = args.out / f"{stem}.jpg"
        # JPEG at a phone-camera quality, so compression artefacts are real too.
        img.convert("RGB").save(path, "JPEG", quality=rng.randint(72, 88))
        meta["condition"] = condition
        (args.out / f"{stem}.expected.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        written += 1
        print(f"  {path.name:<34} {len(meta['drugs'])} drugs, {kind}, {condition}")

    print(f"\n{written} prescriptions -> {args.out}")
    print("Every one is synthetic. Safe to send to a hosted vision model.")


if __name__ == "__main__":
    main()
