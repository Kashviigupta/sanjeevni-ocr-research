"""A tiny, deliberately dumb condition evaluator for ontology `trigger` /
`when` strings like "cc.primary == pain" or "value_hours <= 1 && cc.area == chest".

Supports: ==, <=, >=, <, >, "in [a, b, c]", combined with "&&". Nothing
fancier -- if the ontology ever needs OR, nested groups, or anything else,
that is a real design decision for the ontology's schema, not something to
silently support by writing a bigger parser here.

NOTE: redflags/rules.py has its own near-identical evaluator (written
before this one existed, and already pushed/committed). They should
eventually be unified into one shared module so a fix to one doesn't
silently miss the other -- flagged here rather than done silently, since
redflags/rules.py is already merged and touching it deserves its own
heads-up, not a drive-by change bundled into this file.
"""

from __future__ import annotations

from typing import Any

_OPERATORS = ("<=", ">=", "==", "<", ">", " in ")


class ConditionError(ValueError):
    """Raised when a condition string uses syntax this evaluator doesn't support."""


def condition_matches(condition: str | None, answers: dict[str, Any]) -> bool:
    """True if every '&&'-joined clause in `condition` holds against `answers`.
    A None condition always matches (used for "no trigger = always eligible")."""
    if condition is None:
        return True
    clauses = [c.strip() for c in condition.split("&&")]
    return all(_clause_matches(c, answers) for c in clauses)


def _clause_matches(clause: str, answers: dict[str, Any]) -> bool:
    clause = clause.strip().lstrip("(").rstrip(")").strip()

    for op in _OPERATORS:
        if op in clause:
            left, right = clause.split(op, 1)
            left_val = answers.get(left.strip())
            right = right.strip()
            op = op.strip()

            if op == "in":
                options = [v.strip() for v in right.strip("[]").split(",")]
                return left_val in options
            if op == "==":
                return str(left_val) == right
            try:
                left_num = float(left_val)
                right_num = float(right)
            except (TypeError, ValueError):
                return False
            return {"<=": left_num <= right_num, ">=": left_num >= right_num,
                     "<": left_num < right_num, ">": left_num > right_num}[op]

    raise ConditionError(f"Unsupported condition clause: '{clause}'")
