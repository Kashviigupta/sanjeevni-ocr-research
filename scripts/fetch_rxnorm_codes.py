"""Look up real RxNorm codes for a list of drug names — run this locally, not by Claude.

Why this exists: Claude's sandboxed browsing can only fetch URLs that already
appeared in a search result, so it cannot make live, parameterised API calls.
RxNav needs no API key and no such restriction applies to a normal machine —
this script is the "give someone with real internet access a tool" answer to
that limitation, not a workaround for it.

CRITICAL SAFETY STEP THIS SCRIPT PERFORMS: after finding a candidate RxCUI, it
fetches that concept's own term type (TTY) and REFUSES to accept anything that
isn't TTY=IN (ingredient) or TTY=PIN (precise ingredient). This is exactly the
mistake almost made earlier in this project — a search surfaced the RxCUI for a
cetirizine+pseudoephedrine COMBINATION product when the real target was plain
cetirizine. Filtering by TTY is what prevents that class of error automatically
instead of relying on a human noticing.

Usage:
    pip install requests
    python fetch_rxnorm_codes.py

Output: prints CSV rows to stdout in the same shape as ml/data/drugs_seed.csv
(generic,brands,code,display) — REVIEW EVERY ROW before appending to the real
file. This script finds candidates; a human confirms the drug is one you
actually want in the interaction table before it becomes clinical data.
"""
from __future__ import annotations

import csv
import sys
import time

import requests

# Add whatever's missing from your OPD formulary here. One name per line is
# fine; brand names work too (RxNav normalises "Crocin" the same as "acetaminophen"
# style lookups for the ones RxNorm actually indexes — Indian brand names not in
# RxNorm will simply return no match, which is a correct, honest result).
DRUG_NAMES_TO_LOOKUP = [
    "tizanidine",
]
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
ACCEPTABLE_TTY = {"IN", "PIN"}  # ingredient / precise ingredient only — never a
                                 # combination product, brand pack, or dose form


def find_ingredient_rxcui(name: str) -> tuple[str, str] | None:
    """Return (rxcui, official_name) for the INGREDIENT-level concept, or None.

    Two-step, on purpose: step 1 finds a candidate RxCUI by name; step 2 checks
    that candidate's own TTY before trusting it. Skipping step 2 is exactly how
    a combination-product code gets mistaken for a plain-ingredient one.
    """
    search = requests.get(
        f"{RXNAV_BASE}/rxcui.json", params={"name": name, "search": 1}, timeout=10
    ).json()
    candidates = search.get("idGroup", {}).get("rxnormId", [])
    if not candidates:
        return None

    for rxcui in candidates:
        props = requests.get(
            f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json", timeout=10
        ).json().get("properties", {})
        if props.get("tty") in ACCEPTABLE_TTY:
            return rxcui, props.get("name", name)
    return None  # found something, but nothing at ingredient level — needs a human


def main() -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["generic", "brands", "code", "display"])

    not_found: list[str] = []
    for name in DRUG_NAMES_TO_LOOKUP:
        result = find_ingredient_rxcui(name)
        if result is None:
            not_found.append(name)
            continue
        rxcui, official_name = result
        writer.writerow([official_name.lower(), "", rxcui, official_name])
        time.sleep(0.3)  # be a polite guest of a free public service

    if not_found:
        print(f"# Not found at ingredient level, needs manual lookup: {not_found}", file=sys.stderr)


if __name__ == "__main__":
    main()
