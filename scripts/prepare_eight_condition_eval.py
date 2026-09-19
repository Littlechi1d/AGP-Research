#!/usr/bin/env python3
"""Prepare a new, unscored eight-arm evaluation candidate set.

Labels come only from Facebook graph topology, never from AGP or LLM answers.
The output is a candidate until a human independently reviews its criteria.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_facebook_questions import (  # noqa: E402
    QUESTION_TYPES,
    generate_split,
    normalize,
    read_categories,
    read_graph,
    validate,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain_language_similarity_question(seed_title: str) -> str:
    """Describe shortest-path distance two without graph-specialist jargon."""
    return (
        f"Which pages are the same type as {seed_title} and can be reached "
        "through one other page, but aren't directly connected to it"
    )


def prepare_eval_candidate(
    nodes_path: str | Path,
    edges_path: str | Path,
    targets_path: str | Path,
    excluded_question_paths: list[str | Path],
    output_dir: str | Path,
    *,
    per_type: int = 10,
    random_seed: int = 20260919,
) -> dict:
    """Create new seed-disjoint questions and a blank criteria review form."""
    if per_type < 1:
        raise ValueError("per_type must be positive")
    if not excluded_question_paths:
        raise ValueError("at least one prior question file must be excluded")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    nodes, edges, targets = map(Path, (nodes_path, edges_path, targets_path))
    titles, adjacency = read_graph(nodes, edges)
    categories = read_categories(targets)
    if set(categories) != set(titles):
        raise ValueError("target categories and graph node IDs differ")

    used_seeds: set[str] = set()
    prior_text: set[str] = set()
    prior_ids: set[str] = set()
    excluded_counts: dict[str, int] = {}
    for path_value in excluded_question_paths:
        path = Path(path_value)
        if str(path) in excluded_counts:
            raise ValueError(f"duplicate excluded question file: {path}")
        prior = json.loads(path.read_text(encoding="utf-8"))
        excluded_counts[str(path)] = len(prior)
        for item in prior:
            seeds = item["seed_node_ids"]
            if not seeds or not set(seeds) <= titles.keys():
                raise ValueError(f"invalid prior seeds in {path}: {item['id']}")
            used_seeds.update(seeds)
            prior_text.add(normalize(item["question"]))
            prior_ids.add(item["id"])

    prior_seed_count = len(used_seeds)
    prior_seeds = used_seeds.copy()
    questions = generate_split(
        per_type, titles, adjacency, categories, used_seeds,
        random.Random(random_seed),
    )
    for item in questions:
        if item["question_type"] == "similarity":
            item["question"] = _plain_language_similarity_question(
                titles[item["seed_node_ids"][0]]
            )
    validate(questions, titles, adjacency, categories, per_type)
    for item in questions:
        item["id"] = item["id"].replace("facebook_", "eight_eval_", 1)
    candidate_seeds = {seed for item in questions for seed in item["seed_node_ids"]}
    if prior_seeds & candidate_seeds:
        raise AssertionError("candidate seeds overlap prior questions")
    if any(item["id"] in prior_ids or normalize(item["question"]) in prior_text
           for item in questions):
        raise AssertionError("candidate IDs or question texts overlap prior questions")
    if len({normalize(item["question"]) for item in questions}) != len(questions):
        raise AssertionError("candidate question texts repeat")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-eight-eval-", dir=output.parent) as temp:
        staged = Path(temp)
        questions_file = staged / "questions.json"
        questions_file.write_text(
            json.dumps(questions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with (staged / "criteria_review.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([
                "example_id", "question_type", "question", "seed_titles",
                "expected_answer_titles", "generation_rule", "reviewer_id",
                "question_unambiguous_yes_no", "labels_correct_yes_no", "notes",
            ])
            for item in questions:
                writer.writerow([
                    item["id"], item["question_type"], item["question"],
                    " | ".join(titles[node] for node in item["seed_node_ids"]),
                    " | ".join(titles[node] for node in item["relevant_node_ids"]),
                    item["generation_rule"], "", "", "", "",
                ])
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "candidate_pending_independent_criteria_review",
            "random_seed": random_seed,
            "questions_per_type": per_type,
            "question_count": len(questions),
            "type_counts": dict(Counter(item["question_type"] for item in questions)),
            "question_types": list(QUESTION_TYPES),
            "prior_question_counts": excluded_counts,
            "prior_unique_seed_count": prior_seed_count,
            "candidate_unique_seed_count": len(candidate_seeds),
            "seed_overlap_with_prior": 0,
            "question_text_overlap_with_prior": 0,
            "label_source": "deterministic graph topology; no AGP or LLM outputs used",
            "source_sha256": {
                "nodes": _hash(nodes), "edges": _hash(edges),
                "targets": _hash(targets),
                **{f"excluded:{path}": _hash(Path(path))
                   for path in excluded_question_paths},
            },
            "questions_sha256": _hash(questions_file),
            "criteria_review_sha256": _hash(staged / "criteria_review.csv"),
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--exclude-questions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(prepare_eval_candidate(
        args.nodes, args.edges, args.targets, args.exclude_questions,
        args.output, per_type=args.per_type, random_seed=args.random_seed,
    ), indent=2))


if __name__ == "__main__":
    main()
