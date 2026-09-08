"""Drug-drug interaction checking — KASHVI owns this file.

This is one of the two "killer demo" moments in the pitch: a patient is prescribed a
drug at Clinic B that interacts with one they were given at Hospital A three months ago,
and only a unified longitudinal record can catch it. Nobody holding a paper file can.

Two rules govern this file:

1. **Never claim safety.** We report interactions found in our table. An empty result
   means "nothing in our table", never "this combination is safe". The wording of that
   distinction is a patient-safety matter, not a copy detail.
2. **Normalise to generics first.** Indian prescriptions are written in brand names, and
   the table is keyed on generics. Skip that step and the checker silently finds nothing.
"""
from __future__ import annotations

import logging
from itertools import combinations

from ..schemas import InteractionAlert
from ..terminology.loader import load_interactions
from ..terminology.mapper import generic_name, normalize

log = logging.getLogger(__name__)

SEVERITY_ORDER = {"contraindicated": 0, "major": 1, "moderate": 2, "minor": 3}

DISCLAIMER = (
    "Checked against a curated interaction table. This is not a complete drug "
    "interaction database - always confirm with the prescribing clinician."
)


def _resolve(drug: str) -> str:
    """Brand or generic in, generic out. Falls back to the input when unmatched."""
    return (generic_name(drug) or drug).strip()


def check(drugs: list[str]) -> list[InteractionAlert]:
    """Every pair in the patient's full active medication list.

    Pass the *whole* list — the point of a longitudinal record is catching the pair that
    no single prescriber could see. Checking only the new drug against the new drug
    reproduces exactly the blindness we are trying to fix.
    """
    if len(drugs) < 2:
        return []

    resolved = {normalize(_resolve(d)): d for d in drugs if d and d.strip()}
    table = load_interactions()
    alerts: list[InteractionAlert] = []

    for a_key, b_key in combinations(resolved, 2):
        for row in table:
            pair = {normalize(row.get("drug_a", "")), normalize(row.get("drug_b", ""))}
            if pair == {a_key, b_key}:
                alerts.append(InteractionAlert(
                    severity=row.get("severity", "moderate"),  # type: ignore[arg-type]
                    #: Report the names the patient will recognise from their own strip
                    #: of tablets, not the generic we matched on internally.
                    drug_a=resolved[a_key],
                    drug_b=resolved[b_key],
                    description=row.get("description", "Potential interaction"),
                    source=row.get("source", "curated-table-v1"),
                ))
                break

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.severity, 9))
    log.info("interaction check: %d drugs, %d alerts", len(drugs), len(alerts))  # no names
    return alerts


def summarize(alerts: list[InteractionAlert]) -> str:
    """One plain-language line for the UI.

    Written for a nervous patient, not a pharmacologist — and carefully phrased so that
    "none found" never reads as "safe".
    """
    if not alerts:
        return "No interactions found in our table. " + DISCLAIMER

    worst = alerts[0]
    if worst.severity in ("contraindicated", "major"):
        return (f"Serious interaction: {worst.drug_a} with {worst.drug_b}. "
                f"Speak to your doctor before taking these together.")
    return (f"{len(alerts)} possible interaction(s) found, the most significant between "
            f"{worst.drug_a} and {worst.drug_b}. Mention this at your next visit.")
