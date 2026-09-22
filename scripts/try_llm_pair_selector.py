#!/usr/bin/env python3
"""Test a zero-shot LLM AGP-pair selector using saved development scores only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.eight_condition_retrieval import FIXED_AGP_PAIRS
from agp_research.llm import OpenAICompatibleClient
from agp_research.llm_pair_selector import LLMPairSelector, SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.questions.stem.endswith("_dev"):
        parser.error("only a development questions file ending in _dev.json is allowed")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    score_rows = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line]
    if len(questions) != len(score_rows) or not questions:
        parser.error("questions and saved score rows must be nonempty and aligned")
    for question, row in zip(questions, score_rows):
        if (question["id"], question["question"]) != (row["example_id"], row["question"]):
            parser.error("questions do not match saved development scores")
        if set(row["f1_by_arm"]) != set(FIXED_AGP_PAIRS):
            parser.error("saved scores must cover all five fixed AGP arms")

    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("configure OPENAI_API_KEY and AGP_MODEL in .env")
    selector = LLMPairSelector(client)
    decisions = []
    for question, row in zip(questions, score_rows):
        choice = selector.select(question["question"])
        decisions.append({
            "example_id": question["id"], "question": question["question"],
            "question_type_for_analysis_only": question["question_type"],
            "selected_arm": choice.condition,
            "selected_f1": row["f1_by_arm"][choice.condition],
            "c2_f1": row["f1_by_arm"]["C2"],
            "llm_call": selector.last_call,
        })
        print(f'{question["id"]}: {choice.condition}', flush=True)

    count = len(decisions)
    summary = {
        "question_count": count,
        "selected_arm_counts": dict(sorted(Counter(row["selected_arm"] for row in decisions).items())),
        "llm_selected_mean_f1": sum(row["selected_f1"] for row in decisions) / count,
        "c2_mean_f1": sum(row["c2_f1"] for row in decisions) / count,
        "mean_f1_by_fixed_arm": {
            arm: sum(row["f1_by_arm"][arm] for row in score_rows) / count
            for arm in FIXED_AGP_PAIRS
        },
        "cache_hits": sum(bool(row["llm_call"].get("cache_hit")) for row in decisions),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "selector": "zero-shot question-only LLM pair selection",
        "model": client.model,
        "base_url": client.base_url,
        "temperature": 0,
        "system_prompt": SYSTEM_PROMPT,
        "fixed_pairs": FIXED_AGP_PAIRS,
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.questions, args.scores)
        },
        "warning": "Development-only exploratory comparison; not held-out evidence. V5 wording was previously inspected.",
    }
    args.output.mkdir(parents=True)
    (args.output / "decisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
