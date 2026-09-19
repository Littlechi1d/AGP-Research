#!/usr/bin/env python3
"""Generate eight-condition answers and blinded review artifacts from saved contexts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.eight_condition_answers import (
    DEFAULT_MAX_ANSWER_TOKENS,
    run_eight_condition_answers,
)
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=90055)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-answer-tokens", type=int, default=DEFAULT_MAX_ANSWER_TOKENS,
                        help="Maximum generated tokens per answer (default: 256)")
    args = parser.parse_args()
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("OPENAI_API_KEY is required (any nonempty value works with local Ollama)")
    summary = run_eight_condition_answers(
        client,
        KnowledgeGraph.from_csv(args.nodes, args.edges),
        args.questions, args.contexts, args.output,
        random_seed=args.random_seed,
        max_answer_tokens=args.max_answer_tokens,
        limit=args.limit,
        input_paths={"nodes": args.nodes, "edges": args.edges},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
