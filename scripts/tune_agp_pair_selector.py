#!/usr/bin/env python3
"""Train and validate the question-type AGP pair selector on development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.agp_pair_selector import (
    leave_one_out_validation,
    question_type_from_text,
    train_question_type_selector,
)
from agp_research.agp_native_backend import default_native_library
from agp_research.eight_condition_retrieval import (
    AGP_STUDY_PARAMETERS, FIXED_AGP_PAIRS, NativeAGPBackendPool,
)
from agp_research.evaluation import retrieval_metrics
from agp_research.graph import KnowledgeGraph


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--keyword-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.questions.stem.endswith("_dev"):
        parser.error("selector tuning requires a development questions file ending in _dev.json")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    library = default_native_library()
    if not library.is_file():
        parser.error(f"native AGP library not found: {library}")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    saved_keywords = {
        row["example_id"]: row["keywords"]
        for line in args.keyword_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in (json.loads(line),)
    }
    graph = KnowledgeGraph.from_csv(args.nodes, args.edges)
    rows = []
    with NativeAGPBackendPool(graph) as pool:
        for example in questions:
            example_id = example["id"]
            if example_id not in saved_keywords:
                raise ValueError(f"saved keywords missing for {example_id}")
            predicted_type = question_type_from_text(example["question"])
            if predicted_type != example["question_type"]:
                raise ValueError(f"question wording does not match type for {example_id}")
            seed_ids, unmatched, details = graph.approximate_match(saved_keywords[example_id])
            f1_by_arm = {}
            ranked_by_arm = {}
            for arm, pair in FIXED_AGP_PAIRS.items():
                ranked = pool.query(seed_ids, pair)
                ranked_ids = [item.node.id for item in ranked]
                metrics = retrieval_metrics(ranked_ids, example["relevant_node_ids"])
                f1_by_arm[arm] = _f1(metrics.precision, metrics.recall)
                ranked_by_arm[arm] = ranked_ids
            rows.append(
                {
                    "example_id": example_id,
                    "question": example["question"],
                    "question_type": example["question_type"],
                    "keywords": saved_keywords[example_id],
                    "seed_ids": seed_ids,
                    "unmatched_keywords": unmatched,
                    "keyword_matches": [asdict(item) for item in details],
                    "f1_by_arm": f1_by_arm,
                    "ranked_ids_by_arm": ranked_by_arm,
                }
            )

    selector = train_question_type_selector(rows)
    validation = leave_one_out_validation(rows)
    mean_f1_by_arm = {
        arm: sum(row["f1_by_arm"][arm] for row in rows) / len(rows)
        for arm in FIXED_AGP_PAIRS
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "selector_method": "question template type; highest mean retrieval F1 within type; lowest arm ID breaks ties",
        "fixed_pairs": FIXED_AGP_PAIRS,
        "agp_parameters": asdict(AGP_STUDY_PARAMETERS),
        "query_type": "S",
        "relative_error": 0.10,
        "delta": "1/node_count",
        "native_library_sha256": _hash(library),
        "arm_by_type": selector.arm_by_type,
        "fallback_arm": selector.fallback_arm,
        "mean_f1_by_arm": mean_f1_by_arm,
        "leave_one_out": {key: value for key, value in validation.items() if key != "decisions"},
        "input_sha256": {
            str(path): _hash(path)
            for path in (args.nodes, args.edges, args.questions, args.keyword_results)
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "selector.json").write_text(
        json.dumps(
            {"arm_by_type": selector.arm_by_type, "fallback_arm": selector.fallback_arm},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (args.output / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output / "leave_one_out.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
