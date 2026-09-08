"""Prakriti scoring — **PROVISIONAL, NOT PRACTITIONER REVIEWED.**

Added so a demo can show `dp.prakriti` and `dp.vikriti` computed rather than blank.
The eight assessment axes it reads (build, skin, appetite, digestion, sleep,
temperament, weather tolerance, energy) are the ones published prakriti tools share.
**The thresholds below are not.** Where a dual constitution begins, how close two
doshas must be to count as dual, and whether a simple count is the right model at all
are exactly the judgements a practitioner makes and this module guesses.

Replace before round 2. `git revert` the commit that added this file and the
`prakriti_assessment` ontology section removes the whole thing.

What this module deliberately does NOT do:

* It does not diagnose. Prakriti is constitution — the baseline a person is born with,
  not a disease and not a finding. Nothing here may be shown to a patient as a verdict.
* It does not weight items. A practitioner weights them; a count does not, and
  pretending otherwise would hide the guess inside arithmetic that looks considered.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

DOSHAS = ("vata", "pitta", "kapha")

#: The questions this scoring reads. Kept explicit so a missing item is visible rather
#: than silently reducing the sample the constitution is judged on.
PRAKRITI_ITEMS = (
    "pa.build", "pa.skin", "pa.appetite", "pa.digestion",
    "pa.sleep", "pa.temperament", "pa.weather", "pa.energy",
)

#: How close the second dosha must be to the first to be called dual. PROVISIONAL —
#: a practitioner sets this, and dual constitutions are common enough that the value
#: materially changes how many patients get one.
_DUAL_MARGIN = 1


def score(answers: dict[str, Any]) -> dict[str, int]:
    """Count answers per dosha across the prakriti items."""
    counts = Counter({dosha: 0 for dosha in DOSHAS})
    for item in PRAKRITI_ITEMS:
        value = answers.get(item)
        if isinstance(value, str) and value in DOSHAS:
            counts[value] += 1
    return dict(counts)


def answered_count(answers: dict[str, Any]) -> int:
    return sum(1 for item in PRAKRITI_ITEMS if item in answers)


def derive(answers: dict[str, Any]) -> dict[str, Any] | None:
    """The constitution these answers indicate, or None if too little was answered.

    None is a correct answer, not a failure: a constitution read off two questions is
    not a constitution, and returning a confident-looking one would be worse than
    returning nothing.
    """
    if answered_count(answers) < len(PRAKRITI_ITEMS) - 2:
        return None

    counts = score(answers)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], DOSHAS.index(kv[0])))
    (first, first_n), (second, second_n) = ranked[0], ranked[1]

    dual = (first_n - second_n) <= _DUAL_MARGIN
    return {
        "dominant": f"{first}-{second}" if dual else first,
        "counts": counts,
        "dual": dual,
        "items_answered": answered_count(answers),
        "provisional": True,
        "validation": (
            "PROVISIONAL - scoring not reviewed by an AIIA practitioner. "
            "Not a diagnosis; prakriti is constitution."
        ),
    }
