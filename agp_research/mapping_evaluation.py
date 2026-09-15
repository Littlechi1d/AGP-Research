"""Development-set evaluation helpers for entity-mapping configurations."""

from __future__ import annotations

from typing import Any

from agp_research.graph import KnowledgeGraph


def mapping_metrics(
    graph: KnowledgeGraph,
    examples: list[dict[str, Any]],
    keywords_by_id: dict[str, list[str]],
    *,
    mode: str,
    threshold: float = 0.85,
    margin: float = 0.10,
) -> dict[str, Any]:
    """Evaluate predicted seed-node sets against labelled seed-node sets."""
    true_positive = false_positive = false_negative = exact_sets = 0
    rows: list[dict[str, Any]] = []
    for example in examples:
        example_id = example["id"]
        keywords = keywords_by_id[example_id]
        if mode == "exact":
            predicted, unmatched = graph.exact_match(keywords)
            details: list[dict[str, Any]] = []
        elif mode == "approximate":
            predicted, unmatched, match_details = graph.approximate_match(
                keywords, threshold=threshold, margin=margin
            )
            details = [detail.__dict__ for detail in match_details]
        else:
            raise ValueError(f"Unknown mapping mode: {mode}")

        predicted_set = set(predicted)
        expected_set = set(example["seed_node_ids"])
        true_positive += len(predicted_set & expected_set)
        false_positive += len(predicted_set - expected_set)
        false_negative += len(expected_set - predicted_set)
        exact_sets += predicted_set == expected_set
        rows.append(
            {
                "example_id": example_id,
                "keywords": keywords,
                "expected_seed_ids": sorted(expected_set),
                "predicted_seed_ids": sorted(predicted_set),
                "unmatched_keywords": unmatched,
                "correct_set": predicted_set == expected_set,
                "match_details": details,
            }
        )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "mode": mode,
        "threshold": threshold if mode == "approximate" else None,
        "margin": margin if mode == "approximate" else None,
        "examples": len(examples),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_accuracy": exact_sets / len(examples) if examples else 0.0,
        "exact_sets": exact_sets,
        "rows": rows,
    }


def select_configuration(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer precision, then F1/recall, then conservative larger controls."""
    if not results:
        raise ValueError("At least one mapping result is required")
    return max(
        results,
        key=lambda result: (
            result["precision"],
            result["f1"],
            result["recall"],
            result["threshold"],
            result["margin"],
        ),
    )
