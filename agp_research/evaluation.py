"""Retrieval evaluation and batch experiment runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agp_research.models import AGPParameters
from agp_research.pipeline import AGPPipeline


@dataclass(frozen=True)
class RetrievalMetrics:
    precision: float
    recall: float
    hit_rate: float
    reciprocal_rank: float


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str]) -> RetrievalMetrics:
    relevant = set(relevant_ids)
    retrieved = list(dict.fromkeys(ranked_ids))
    hits = sum(node_id in relevant for node_id in retrieved)
    first_rank = next((rank for rank, node_id in enumerate(retrieved, 1) if node_id in relevant), None)
    return RetrievalMetrics(
        precision=hits / len(retrieved) if retrieved else 0.0,
        recall=hits / len(relevant) if relevant else 0.0,
        hit_rate=float(hits > 0),
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
    )


def run_experiment(
    pipeline: AGPPipeline,
    questions_path: str | Path,
    output_path: str | Path,
    strategies: list[str],
) -> dict[str, dict[str, float]]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = {strategy: {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "reciprocal_rank": 0.0, "latency": 0.0} for strategy in strategies}
    counts = {strategy: 0 for strategy in strategies}
    with output.open("w", encoding="utf-8") as stream:
        for example in questions:
            for strategy in strategies:
                result = pipeline.run(
                    example["question"],
                    strategy=strategy,
                    fixed_parameters=AGPParameters(depth=2, decay=0.6, top_k=10),
                    generate_answer=False,
                )
                ranked_ids = [item.node.id for item in result.ranked_nodes]
                metrics = retrieval_metrics(ranked_ids, example["relevant_node_ids"])
                record = result.to_dict() | {"example_id": example["id"], "metrics": metrics.__dict__}
                stream.write(json.dumps(record) + "\n")
                counts[strategy] += 1
                for key in ("precision", "recall", "hit_rate", "reciprocal_rank"):
                    totals[strategy][key] += getattr(metrics, key)
                totals[strategy]["latency"] += result.elapsed_seconds
    return {
        strategy: {key: value / counts[strategy] for key, value in values.items()}
        for strategy, values in totals.items()
    }

