"""Load the curated terminology tables — KASHVI owns this file.

Why CSVs instead of a terminology server: SNOMED CT requires a licence, UMLS requires
registration, and a hackathon demo must run offline on a laptop with no network. A
hand-checked table of the terms that actually appear on Indian lab reports and
prescriptions beats an API we cannot ship.

Every row is hand-checked. A wrong code here becomes a wrong code in a patient's
permanent record, so treat additions to `data/*.csv` as clinical work, not data entry.
"""
from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        log.warning("terminology table missing: %s", path.name)
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [
            {k.strip(): (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(fh)
            if row and not (row.get("term") or "").startswith("#")
        ]


@lru_cache
def load_lab_terms() -> list[dict[str, str]]:
    """`term,synonyms,system,code,display,unit` — analytes to LOINC.

    `synonyms` is pipe-separated and does the heavy lifting: one report says
    "FBS", another "Fasting Blood Sugar", a third "Glucose, Fasting".
    """
    return _read(settings.data_dir / "loinc_seed.csv")


@lru_cache
def load_drugs() -> list[dict[str, str]]:
    """`generic,brands,system,code,display,class` — brand and generic drug names.

    Indian prescriptions are written in brand names almost exclusively, so `brands`
    is not optional decoration; it is the primary lookup key.
    """
    return _read(settings.data_dir / "drugs_seed.csv")


@lru_cache
def load_interactions() -> list[dict[str, str]]:
    """`drug_a,drug_b,severity,description,source` — generic names only.

    Normalise brands to generics before checking, or the table silently misses.
    """
    return _read(settings.data_dir / "interactions_seed.csv")


@lru_cache
def load_conditions() -> list[dict[str, str]]:
    """`term,synonyms,system,code,display` — diagnoses to SNOMED CT / ICD-11."""
    return _read(settings.data_dir / "snomed_seed.csv")


def term_count() -> int:
    """Reported by /health so the team can see the tables actually loaded."""
    return sum(len(t) for t in (load_lab_terms(), load_drugs(), load_conditions()))


def reload_tables() -> int:
    """Drop the caches after editing a CSV, without restarting the service."""
    for fn in (load_lab_terms, load_drugs, load_interactions, load_conditions):
        fn.cache_clear()
    return term_count()
