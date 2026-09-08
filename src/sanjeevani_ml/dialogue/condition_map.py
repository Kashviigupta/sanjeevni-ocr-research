"""Condition-map reasoning for dialogue question ordering.

This module does NOT diagnose the patient.

It only:
1. updates condition plausibility from recorded answers
2. identifies unanswered questions that distinguish live conditions
3. ranks those questions for dialogue ordering

Scores are additive and clamped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Condition:
    id: str
    urgency: str | None
    discriminators: tuple[dict[str, Any], ...]
    negative: tuple[dict[str, Any], ...]
    must_ask: tuple[str, ...]


class ConditionMap:
    def __init__(self, path: str):
        self.path = Path(path)

        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.conditions = [
            Condition(
                id=item["id"],
                urgency=item.get("urgency"),
                discriminators=tuple(
                    item.get("discriminators", [])
                ),
                negative=tuple(
                    item.get("negative", [])
                ),
                must_ask=tuple(
                    item.get("must_ask", [])
                ),
            )
            for item in raw.get("conditions", [])
        ]

    @staticmethod
    def _value_matches(
        answer_value: Any,
        expected_value: Any,
    ) -> bool:
        """Handle scalar and multi-select answers."""

        if isinstance(answer_value, (list, tuple, set)):
            return expected_value in answer_value

        return answer_value == expected_value

    def calculate_plausibility(
        self,
        answers: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate plausibility for every condition.

        Score =
            positive matched weights
            - negative matched weights

        Final score is clamped to [0, 1].
        """

        scores: dict[str, float] = {}

        for condition in self.conditions:
            score = 0.0

            # Positive evidence
            for discriminator in condition.discriminators:
                question_id = discriminator["question_id"]
                expected = discriminator["value"]
                weight = float(discriminator["weight"])

                if question_id not in answers:
                    continue

                if self._value_matches(
                    answers[question_id],
                    expected,
                ):
                    score += weight

            # Negative evidence
            for negative in condition.negative:
                question_id = negative["question_id"]
                expected = negative["value"]
                weight = float(negative["weight"])

                if question_id not in answers:
                    continue

                if self._value_matches(
                    answers[question_id],
                    expected,
                ):
                    score -= weight

            scores[condition.id] = max(
                0.0,
                min(1.0, score),
            )

        return scores

    def _question_profile(
        self,
        question_id: str,
        condition: Condition,
    ) -> list[float]:
        """Return the possible effect of a question on a condition."""

        effects: list[float] = []

        for item in condition.discriminators:
            if item["question_id"] == question_id:
                effects.append(float(item["weight"]))

        for item in condition.negative:
            if item["question_id"] == question_id:
                effects.append(-float(item["weight"]))

        return effects

    def question_score(
        self,
        question_id: str,
        live_conditions: list[Condition],
        plausibility: dict[str, float],
    ) -> float:
        """Score how useful a question is for separating live conditions."""

        if not live_conditions:
            return 0.0

        # Must-ask questions have highest priority.
        for condition in live_conditions:
            if question_id in condition.must_ask:
                return 100.0

        # If only one condition is live, prefer questions that
        # provide evidence for or against that condition.
        if len(live_conditions) == 1:
            effects = self._question_profile(
                question_id,
                live_conditions[0],
            )

            return max(
                (abs(effect) for effect in effects),
                default=0.0,
            )

        # Multiple conditions are live.
        # Prefer questions whose effects differ between them.
        condition_effects: list[tuple[float, float]] = []

        for condition in live_conditions:
            effects = self._question_profile(
                question_id,
                condition,
            )

            if effects:
                effect = max(effects, key=abs)
            else:
                effect = 0.0

            condition_effects.append(
                (
                    plausibility.get(condition.id, 0.0),
                    effect,
                )
            )

        separation = 0.0

        for i in range(len(condition_effects)):
            p_i, effect_i = condition_effects[i]

            for j in range(i + 1, len(condition_effects)):
                p_j, effect_j = condition_effects[j]

                separation += (
                    max(p_i, p_j)
                    * abs(effect_i - effect_j)
                )

        return separation

    def rank_questions(
        self,
        candidates: list[str],
        answers: dict[str, Any],
    ) -> list[str]:
        """Rank unanswered questions using the condition map.

        If there is no condition evidence yet, the original ontology
        order is preserved.
        """

        plausibility = self.calculate_plausibility(
            answers
        )

        live_conditions = [
            condition
            for condition in self.conditions
            if plausibility.get(condition.id, 0.0) > 0.0
        ]

        # No condition evidence yet.
        # Keep the ontology's original order.
        if not live_conditions:
            return candidates

        urgency_rank = {
            "critical": 3,
            "urgent": 2,
            "soon": 1,
            None: 0,
        }

        def sort_key(question_id: str):
            score = self.question_score(
                question_id,
                live_conditions,
                plausibility,
            )

            # Urgent must-ask questions win ties.
            required_urgency = max(
                (
                    urgency_rank.get(
                        condition.urgency,
                        0,
                    )
                    for condition in live_conditions
                    if question_id in condition.must_ask
                ),
                default=0,
            )

            return (
                score,
                required_urgency,
            )

        # Python's sort is stable, so questions with equal
        # scores retain their original ontology order.
        return sorted(
            candidates,
            key=sort_key,
            reverse=True,
        )

    def plausibility(
        self,
        answers: dict[str, Any],
    ) -> dict[str, float]:
        """Return current condition plausibility."""

        return self.calculate_plausibility(answers)

    def live_conditions(
        self,
        answers: dict[str, Any],
    ) -> list[str]:
        """Return conditions with non-zero plausibility."""

        scores = self.calculate_plausibility(answers)

        return [
            condition.id
            for condition in self.conditions
            if scores.get(condition.id, 0.0) > 0.0
        ]

    def must_ask_for_live_conditions(
        self,
        answers: dict[str, Any],
    ) -> set[str]:
        """Return must-ask questions for currently live conditions."""

        scores = self.calculate_plausibility(answers)

        result: set[str] = set()

        for condition in self.conditions:
            if scores.get(condition.id, 0.0) <= 0.0:
                continue

            result.update(condition.must_ask)

        return result