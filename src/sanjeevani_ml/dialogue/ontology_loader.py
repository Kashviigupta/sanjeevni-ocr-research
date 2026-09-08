"""Loads shared/clinical/history-ontology.json — the REAL nested structure
(sections -> questions -> options), not a flat list.

This replaces an earlier version that assumed a flat top-level "questions"
array. That version was written before the real ontology content was seen
and never matched it -- rewritten here against the actual file.
"""

from __future__ import annotations

import json
from pathlib import Path


class OntologyError(RuntimeError):
    """Raised when the ontology file is missing or structurally invalid."""


#: Sections key their items under "questions" — except `dashavidha_pariksha`, which
#: uses "parameters" because the ten-fold examination is a set of named parameters,
#: not a list of plain questions. Structurally they are the same thing, so every
#: reader goes through `section_items` rather than reaching for a key directly.
_ITEM_KEYS = ("questions", "parameters")

#: `derived` items are COMPUTED, never asked: dp.prakriti is a scored
#: sub-questionnaire, dp.vikriti comes from complaint + prakriti, dp.vaya from the
#: date of birth. Putting them in the askable set would have the kiosk ask a patient
#: their own Vikriti, which is not a question a person can answer.
_DERIVED_INPUT_TYPE = "derived"

#: Fallback used when the ontology file predates `interaction_defaults` --
#: an older file should still load, just with this default, not crash.
_DEFAULT_INTERACTION_DEFAULTS = {"tap_timeout_seconds": 10}


def section_items(section: dict, *, include_derived: bool = False) -> list[dict]:
    """Every askable item in a section, whichever key it uses.

    Derived items are excluded by default — they are computed from other answers,
    so they are not questions the interview can put to a patient. Pass
    `include_derived=True` when you need the full declared set (e.g. to report
    which parameters a summary still owes a value for).
    """
    items = next((section[key] for key in _ITEM_KEYS if key in section), [])
    if include_derived:
        return list(items)
    return [item for item in items if item.get("input_type") != _DERIVED_INPUT_TYPE]


def load_ontology(path: str | Path) -> dict:
    """Read and minimally validate the ontology JSON. Returns the raw dict --
    callers (engine.py, redflags/rules.py) work with sections/questions
    directly, since red_flag/trigger/branch metadata doesn't fit a trimmed
    Question schema.
    """
    path = Path(path)
    if not path.exists():
        raise OntologyError(f"Ontology file not found: {path}")

    try:
        ontology = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OntologyError(f"Ontology file is not valid JSON: {exc}") from exc

    sections = ontology.get("sections")
    if not isinstance(sections, list) or not sections:
        raise OntologyError("Ontology file has no 'sections' list, or it's empty.")

    seen_question_ids: set[str] = set()
    for section in sections:
        if "id" not in section:
            raise OntologyError(f"Section has no 'id': {sorted(section)}")
        if not any(key in section for key in _ITEM_KEYS):
            # Report the section id, not the whole section: these dicts are large
            # enough that dumping one buries the actual problem.
            raise OntologyError(
                f"Section '{section['id']}' has neither 'questions' nor 'parameters'"
            )
        # include_derived: a derived item still owns its id, so it must take part in
        # duplicate detection even though it is never asked.
        for question in section_items(section, include_derived=True):
            qid = question.get("id")
            if not qid:
                raise OntologyError(f"Item in section '{section['id']}' has no 'id'")
            if qid in seen_question_ids:
                raise OntologyError(f"Duplicate question id across ontology: '{qid}'")
            seen_question_ids.add(qid)

    return ontology


def flatten_questions(ontology: dict) -> dict[str, dict]:
    """{question_id: raw entry} for every ASKABLE item, whichever section it is in.

    Derived items are absent by design — nothing that walks this map should be able
    to put one to a patient.
    """
    flat: dict[str, dict] = {}
    for section in ontology.get("sections", []):
        for question in section_items(section):
            flat[question["id"]] = question
    return flat


def derived_item_ids(ontology: dict) -> list[str]:
    """Ids the ontology declares but the interview never asks.

    The physician's summary should be able to say these were computed rather than
    elicited, and `ExplorationReport.not_explored` should be able to name any that
    could not be computed at all.
    """
    derived: list[str] = []
    for section in ontology.get("sections", []):
        askable = {item["id"] for item in section_items(section)}
        for item in section_items(section, include_derived=True):
            if item.get("id") and item["id"] not in askable:
                derived.append(item["id"])
    return derived


def section_for_question(ontology: dict, question_id: str) -> dict | None:
    for section in ontology.get("sections", []):
        for question in section_items(section):
            if question["id"] == question_id:
                return section
    return None


def get_interaction_defaults(ontology: dict) -> dict:
    """Expose the ontology's top-level `interaction_defaults` block.

    Currently just `{tap_timeout_seconds: int}` -- how long a client waits with
    no answer in tap-first mode before auto-opening the mic. This is NOT session
    state and NOT per-question: the patient can switch between speak-only and
    tap-first at any question, and neither client nor engine tracks which mode is
    "current" anywhere. This is the one number both kiosk clients need to agree
    on, so it lives here instead of being hardcoded separately in Meet's Flutter
    app and Samridhi's web kiosk.

    Falls back to a sane default rather than raising if an older ontology file
    predates this key -- a missing UX-timing default should never be a hard
    failure the way a missing 'sections' list is.
    """
    return ontology.get("interaction_defaults", _DEFAULT_INTERACTION_DEFAULTS)
