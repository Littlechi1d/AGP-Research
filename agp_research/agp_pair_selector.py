"""Question-only selection of one predeclared native AGP `(a, b)` pair."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agp_research.eight_condition_retrieval import FIXED_AGP_PAIRS
from agp_research.graph import normalize


QUESTION_TYPES = ("direct", "comparison", "path", "similarity")


def question_type_from_text(question: str) -> str | None:
    """Recognize the four benchmark templates without reading gold labels."""
    wording = normalize(question)
    if "directly neighbor" in wording:
        return "direct"
    if "liked by both" in wording:
        return "comparison"
    if "shortest path between" in wording:
        return "path"
    if "similar to" in wording and "two hops away" in wording:
        return "similarity"
    return None


@dataclass(frozen=True)
class PairSelection:
    condition: str
    a: float
    b: float
    question_type: str | None
    used_fallback: bool


@dataclass(frozen=True)
class QuestionTypePairSelector:
    """Map a question's wording to a development-selected fixed AGP arm."""

    arm_by_type: dict[str, str]
    fallback_arm: str

    def __post_init__(self) -> None:
        if set(self.arm_by_type) != set(QUESTION_TYPES):
            raise ValueError("selector must configure all four question types")
        for arm in (*self.arm_by_type.values(), self.fallback_arm):
            if arm not in FIXED_AGP_PAIRS:
                raise ValueError(f"unknown AGP arm: {arm}")

    def select(self, question: str) -> PairSelection:
        """Select from wording only—never from test labels or AGP scores."""
        question_type = question_type_from_text(question)
        arm = self.arm_by_type.get(question_type, self.fallback_arm)
        a, b = FIXED_AGP_PAIRS[arm]
        return PairSelection(arm, a, b, question_type, question_type is None)

    @classmethod
    def from_json(cls, path: str | Path) -> "QuestionTypePairSelector":
        """Load and validate the saved development selector for later runs."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["arm_by_type"], data["fallback_arm"])


def best_fixed_arm(rows: list[dict[str, Any]]) -> str:
    """Maximize development mean retrieval F1; tie-break to lowest arm ID."""
    if not rows:
        raise ValueError("at least one development row is required")
    for row in rows:
        if set(row["f1_by_arm"]) != set(FIXED_AGP_PAIRS):
            raise ValueError("each row must score all five fixed AGP arms")
    means = {
        arm: sum(row["f1_by_arm"][arm] for row in rows) / len(rows)
        for arm in FIXED_AGP_PAIRS
    }
    return min(FIXED_AGP_PAIRS, key=lambda arm: (-means[arm], arm))


def train_question_type_selector(rows: list[dict[str, Any]]) -> QuestionTypePairSelector:
    """Fit one arm per type using labelled development rows only."""
    if not rows:
        raise ValueError("at least one development row is required")
    arm_by_type = {}
    fallback = best_fixed_arm(rows)
    for question_type in QUESTION_TYPES:
        typed = [row for row in rows if row["question_type"] == question_type]
        arm_by_type[question_type] = best_fixed_arm(typed) if typed else fallback
    return QuestionTypePairSelector(arm_by_type, fallback)


def leave_one_out_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate selector choices with each development question held out once."""
    if len(rows) < 2:
        raise ValueError("leave-one-out validation requires at least two rows")
    decisions = []
    for heldout_index, heldout in enumerate(rows):
        training = rows[:heldout_index] + rows[heldout_index + 1 :]
        selector = train_question_type_selector(training)
        choice = selector.select(heldout["question"])
        if choice.question_type != heldout["question_type"]:
            raise ValueError(f"question wording does not match label for {heldout['example_id']}")
        global_arm = best_fixed_arm(training)
        decisions.append(
            {
                "example_id": heldout["example_id"],
                "question_type": heldout["question_type"],
                "selected_arm": choice.condition,
                "selected_f1": heldout["f1_by_arm"][choice.condition],
                "global_arm": global_arm,
                "global_f1": heldout["f1_by_arm"][global_arm],
            }
        )
    return {
        "question_count": len(decisions),
        "adaptive_mean_f1": sum(row["selected_f1"] for row in decisions) / len(decisions),
        "global_mean_f1": sum(row["global_f1"] for row in decisions) / len(decisions),
        "decisions": decisions,
    }
