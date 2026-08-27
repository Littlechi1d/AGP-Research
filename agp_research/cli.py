"""Command-line interface for demos and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agp_research.evaluation import run_experiment
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient
from agp_research.models import AGPParameters
from agp_research.paper_backend import PaperAGPBackend, PaperBackendConfig
from agp_research.pipeline import AGPPipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Query-adaptive graph propagation prototype")
    result.add_argument("--nodes", default="data/nodes.csv")
    result.add_argument("--edges", default="data/edges.csv")
    result.add_argument("--backend", choices=["python", "paper"], default="python")
    result.add_argument("--paper-executable", default="build/paper_backend/agp_query_bridge")
    result.add_argument("--paper-a", type=float, default=0.0)
    result.add_argument("--paper-b", type=float, default=1.0)
    result.add_argument("--paper-delta", type=float)
    result.add_argument("--paper-relative-error", type=float, default=0.1)
    result.add_argument("--paper-query-type", choices=["N", "S"], default="S")
    commands = result.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Run one question")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=["seed-only", "fixed", "rules", "llm"], default="rules")
    ask.add_argument("--depth", type=int, default=2)
    ask.add_argument("--decay", type=float, default=0.6)
    ask.add_argument("--top-k", type=int, default=10)
    ask.add_argument("--no-answer", action="store_true")
    experiment = commands.add_parser("experiment", help="Compare retrieval strategies")
    experiment.add_argument("--questions", default="data/questions.json")
    experiment.add_argument("--output", default="results/experiment.jsonl")
    experiment.add_argument("--strategies", nargs="+", default=["seed-only", "fixed", "rules"])
    return result


def main() -> None:
    args = parser().parse_args()
    graph = KnowledgeGraph.from_csv(args.nodes, args.edges)
    backend = None
    if args.backend == "paper":
        backend = PaperAGPBackend(
            Path(args.paper_executable),
            PaperBackendConfig(
                a=args.paper_a,
                b=args.paper_b,
                delta=args.paper_delta,
                relative_error=args.paper_relative_error,
                query_type=args.paper_query_type,
            ),
        )
    pipeline = AGPPipeline(
        graph,
        OpenAICompatibleClient.from_environment(),
        propagation_backend=backend,
    )
    if args.command == "ask":
        result = pipeline.run(
            args.question,
            strategy=args.strategy,
            fixed_parameters=AGPParameters(args.depth, args.decay, args.top_k),
            generate_answer=not args.no_answer,
        )
        print(json.dumps(result.to_dict(), indent=2))
    else:
        summary = run_experiment(pipeline, args.questions, args.output, args.strategies)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
