#!/usr/bin/env python3
"""Evaluate the question-only LLM pair selector on saved development contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.eight_condition_retrieval import FIXED_AGP_PAIRS
from agp_research.llm import OpenAICompatibleClient
from agp_research.llm_pair_selector import LLMPairSelector, SYSTEM_PROMPT


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrieval_f1(predicted: list[str], relevant: list[str]) -> float:
    predicted_set, relevant_set = set(predicted), set(relevant)
    overlap = len(predicted_set & relevant_set)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(relevant_set) if relevant_set else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(
    client: OpenAICompatibleClient,
    questions_path: str | Path,
    contexts_path: str | Path,
    output_dir: str | Path,
) -> dict:
    questions_file = Path(questions_path)
    contexts_file = Path(contexts_path)
    output = Path(output_dir)
    if not questions_file.stem.endswith("_dev"):
        raise ValueError("LLM selector development evaluation requires questions_dev.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("development questions must not be empty")

    # Make all question-only choices before opening contexts or relevance labels.
    selector = LLMPairSelector(client)
    selections = {}
    for item in questions:
        choice = selector.select(item["question"])
        selections[item["id"]] = {
            "selected_arm": choice.condition,
            "a": choice.a,
            "b": choice.b,
            "llm_call": dict(selector.last_call or {}),
        }
        print(f'{item["id"]}: {choice.condition}', flush=True)

    context_manifest_path = contexts_file.parent / "manifest.json"
    context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    if context_manifest.get("contexts_sha256") != _hash(contexts_file):
        raise ValueError("saved contexts do not match their manifest checksum")
    if context_manifest.get("input_sha256", {}).get("questions") != _hash(questions_file):
        raise ValueError("questions do not match the saved context manifest")
    context_rows = {
        row["example_id"]: row
        for line in contexts_file.read_text(encoding="utf-8").splitlines() if line
        for row in (json.loads(line),)
    }
    if set(context_rows) != set(selections):
        raise ValueError("questions and contexts contain different example IDs")

    rows = []
    arm_totals = Counter({arm: 0.0 for arm in FIXED_AGP_PAIRS})
    selected_total = oracle_total = 0.0
    for item in questions:
        context = context_rows[item["id"]]
        if context["question"] != item["question"]:
            raise ValueError(f"question mismatch for {item['id']}")
        f1_by_arm = {
            arm: _retrieval_f1(
                context["conditions"][arm]["ranked_node_ids"],
                item["relevant_node_ids"],
            )
            for arm in FIXED_AGP_PAIRS
        }
        selected_arm = selections[item["id"]]["selected_arm"]
        selected_f1 = f1_by_arm[selected_arm]
        oracle_f1 = max(f1_by_arm.values())
        selected_total += selected_f1
        oracle_total += oracle_f1
        arm_totals.update(f1_by_arm)
        rows.append({
            "example_id": item["id"],
            "question": item["question"],
            "question_type_for_analysis_only": item["question_type"],
            **selections[item["id"]],
            "selected_f1": selected_f1,
            "f1_by_arm": f1_by_arm,
            "oracle_f1": oracle_f1,
        })

    count = len(rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "question_count": count,
        "model": client.model,
        "base_url": client.base_url,
        "selection_counts": dict(sorted(Counter(
            row["selected_arm"] for row in rows
        ).items())),
        "llm_selected_mean_retrieval_f1": selected_total / count,
        "mean_retrieval_f1_by_fixed_arm": {
            arm: arm_totals[arm] / count for arm in FIXED_AGP_PAIRS
        },
        "oracle_mean_retrieval_f1": oracle_total / count,
        "system_prompt": SYSTEM_PROMPT,
        "selection_completed_before_context_access": True,
        "input_sha256": {
            "questions": _hash(questions_file),
            "contexts": _hash(contexts_file),
            "context_manifest": _hash(context_manifest_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-llm-selector-dev-", dir=output.parent) as temp:
        staged = Path(temp)
        (staged / "decisions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (staged / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("configure the local LLM in .env")
    print(json.dumps(evaluate(
        client, args.questions, args.contexts, args.output
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
