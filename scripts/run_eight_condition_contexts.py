#!/usr/bin/env python3
"""Save C0–C7 retrieval contexts; do not generate answers yet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.agp_native_backend import default_native_library
from agp_research.agp_pair_selector import QuestionTypePairSelector
from agp_research.eight_condition_contexts import run_eight_condition_contexts
from agp_research.eight_condition_retrieval import NativeAGPBackendPool
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keyword-results", type=Path,
                        help="Use saved LLM keywords; otherwise call the configured model once per question")
    parser.add_argument("--native-library", type=Path, default=default_native_library())
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    client = None if args.keyword_results else OpenAICompatibleClient.from_environment()
    if args.keyword_results is None and client is None:
        parser.error("OPENAI_API_KEY is required unless --keyword-results is supplied")
    graph = KnowledgeGraph.from_csv(args.nodes, args.edges)
    selector = QuestionTypePairSelector.from_json(args.selector)
    with NativeAGPBackendPool(graph, args.native_library) as pool:
        manifest = run_eight_condition_contexts(
            graph, args.questions, args.output, selector, pool,
            client=client,
            keyword_results_path=args.keyword_results,
            input_paths={
                "nodes": args.nodes, "edges": args.edges,
                "selector": args.selector, "native_library": args.native_library,
            },
            limit=args.limit,
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
