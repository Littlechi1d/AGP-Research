#!/usr/bin/env python3
"""Run the predeclared frozen conditions without overwriting prior evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.frozen_evaluation import run_frozen_evaluation
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient
from agp_research.pipeline import AGPPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("OPENAI_API_KEY is required for the frozen LLM conditions")
    pipeline = AGPPipeline(KnowledgeGraph.from_csv(args.nodes, args.edges), client)
    summary = run_frozen_evaluation(pipeline, args.questions, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
