#!/usr/bin/env python3
"""Generate a paired LLM-only versus AGP-grounded answer study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.answer_evaluation import run_answer_quality_study
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--retrieval-strategy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=90055)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("OPENAI_API_KEY is required (use any non-empty value for local Ollama)")
    summary = run_answer_quality_study(
        client,
        KnowledgeGraph.from_csv(args.nodes, args.edges),
        args.questions,
        args.retrieval_results,
        args.output,
        retrieval_strategy=args.retrieval_strategy,
        random_seed=args.random_seed,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
