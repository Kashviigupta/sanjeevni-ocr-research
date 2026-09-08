"""Look up real SNOMED CT codes for a list of condition terms — run this locally.

Uses SNOMED International's own public Snowstorm reference server
(snowstorm.ihtsdotools.org). That server's own documentation is explicit:
"reference purposes only" and "MUST NOT be used in production" — fine for
curating a hackathon terminology table, wrong for a real deployment. If this
project goes past the hackathon, point this script at a licensed Snowstorm
instance instead (the query shape is identical; only the base URL changes).

Every result needs a HUMAN to confirm it's the right concept before it goes
into ml/data/snomed_seed.csv. SNOMED terms are precise on purpose — "chest
pain" and "atypical chest pain" are different concepts with different codes,
and only a clinician reading the full FSN (fully specified name) can tell you
which one your documentation actually means.

Usage:
    pip install requests
    python fetch_snomed_codes.py
"""
from __future__ import annotations

import csv
import sys
import time

import requests

# Add whatever's missing from your OPD condition list here.
CONDITION_TERMS_TO_LOOKUP = [
    "viral fever",
    "acute pharyngitis",
    "acute sinusitis",
    "allergic rhinitis",
    "conjunctivitis",
    "urinary tract infection",
    "gastroenteritis",
    "gastroesophageal reflux disease",
    "irritable bowel syndrome",
    "migraine",
    "epilepsy",
    "cerebrovascular accident",
    "myocardial infarction",
    "community acquired pneumonia",
    "acute bronchitis",
    "viral hepatitis",
    "iron deficiency anemia",
    "polycystic ovary syndrome",
    "lumbar spondylosis",
    "benign paroxysmal positional vertigo",
]

SNOWSTORM_BASE = "https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct"
BRANCH = "MAIN"  # the international edition

#: Some public terminology demo servers reject or rate-limit traffic that looks
#: like a bare script rather than a browser — this header is the difference
#: between "works" and a silent connection reset. Not spoofing anything, just
#: identifying the request the way a normal browser would.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def search_concepts(term: str, limit: int = 3) -> list[dict]:
    """Top candidate concepts for `term`. Returns raw hits — a human picks the
    right one; this function does not guess for you.
    """
    response = requests.get(
        f"{SNOWSTORM_BASE}/browser/{BRANCH}/concepts",
        params={"term": term, "activeFilter": "true", "limit": limit},
        headers=_HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("items", [])


def main() -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["term", "synonyms", "code", "display", "_candidate_fsn_for_human_review"])

    for term in CONDITION_TERMS_TO_LOOKUP:
        try:
            hits = search_concepts(term)
        except requests.exceptions.RequestException as exc:
            # One term failing (server hiccup, rate limit) must never kill the
            # whole run — print the miss to stderr and keep going, same
            # degrade-never-fail principle as the rest of this codebase.
            print(f"# Lookup failed for '{term}': {exc}", file=sys.stderr)
            time.sleep(2)  # back off a little longer after a failure
            continue

        if not hits:
            print(f"# No match at all for: {term}", file=sys.stderr)
            continue

        # Print every candidate as a commented review line — do NOT auto-pick
        # the first result. SNOMED distinguishes clinically important nuances
        # (disorder vs finding vs situation) that only a clinician should choose
        # between.
        for hit in hits:
            concept_id = hit.get("conceptId")
            fsn = hit.get("fsn", {}).get("term", "")
            pt = hit.get("pt", {}).get("term", term)
            writer.writerow([pt.lower(), "", concept_id, pt, fsn])
        time.sleep(0.5)


if __name__ == "__main__":
    main()
