"""One-time frozen evaluation runner and deterministic statistical summaries."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agp_research.evaluation import retrieval_metrics
from agp_research.models import AGPParameters
from agp_research.pipeline import AGPPipeline


BOOTSTRAP_SEED = 90055
BOOTSTRAP_SAMPLES = 10_000
FROZEN_MODEL_DIGEST = (
    "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
)


@dataclass(frozen=True)
class FrozenCondition:
    name: str
    strategy: str
    parameters: AGPParameters | None = None


FROZEN_CONDITIONS = (
    FrozenCondition("seed-only", "seed-only"),
    FrozenCondition("fixed-k5", "fixed", AGPParameters(2, 0.3, 5)),
    FrozenCondition("fixed-k10", "fixed", AGPParameters(2, 0.3, 10)),
    FrozenCondition("rules", "rules"),
    FrozenCondition("llm-keywords", "llm-keywords", AGPParameters(2, 0.3, 5)),
    FrozenCondition("llm-parameters", "llm-parameters"),
    FrozenCondition("llm-parameters-few-shot", "llm-parameters-few-shot"),
    FrozenCondition("llm", "llm"),
)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for condition in dict.fromkeys(record["condition"] for record in records):
        selected = [record for record in records if record["condition"] == condition]
        result[condition] = {
            metric: _mean([record["metrics"][metric] for record in selected])
            for metric in (
                "precision",
                "recall",
                "f1",
                "hit_rate",
                "reciprocal_rank",
            )
        }
        result[condition]["latency"] = _mean(
            [record["elapsed_seconds"] for record in selected]
        )
    return result


def _summarize_by_type(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for question_type in dict.fromkeys(record["question_type"] for record in records):
        selected = [
            record for record in records if record["question_type"] == question_type
        ]
        result[question_type] = _summarize(selected)
    return result


def _paired_bootstrap(
    records: list[dict[str, Any]],
    *,
    baseline: str = "rules",
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for record in records:
        scores.setdefault(record["condition"], {})[record["example_id"]] = record[
            "metrics"
        ]["f1"]
    question_ids = list(scores[baseline])
    rng = random.Random(seed)
    result = {}
    for condition, condition_scores in scores.items():
        if condition == baseline:
            continue
        differences = [
            condition_scores[question_id] - scores[baseline][question_id]
            for question_id in question_ids
        ]
        bootstrap_means = sorted(
            _mean([differences[rng.randrange(len(differences))] for _ in differences])
            for _ in range(samples)
        )
        lower_index = int(0.025 * (samples - 1))
        upper_index = int(0.975 * (samples - 1))
        result[condition] = {
            "baseline": baseline,
            "mean_f1_difference": _mean(differences),
            "ci95_lower": bootstrap_means[lower_index],
            "ci95_upper": bootstrap_means[upper_index],
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_frozen_evaluation(
    pipeline: AGPPipeline,
    questions_path: str | Path,
    output_dir: str | Path,
    *,
    conditions: tuple[FrozenCondition, ...] = FROZEN_CONDITIONS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Run every named condition once and write a complete result bundle."""
    source = Path(questions_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    questions = json.loads(source.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("question file must contain at least one question")
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique")
    if not any(condition.name == "rules" for condition in conditions):
        raise ValueError("conditions must include the rules bootstrap baseline")

    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    output_path = destination / "results.jsonl"
    with output_path.open("x", encoding="utf-8") as stream:
        for example in questions:
            for condition in conditions:
                result = pipeline.run(
                    example["question"],
                    strategy=condition.strategy,
                    fixed_parameters=condition.parameters,
                    generate_answer=False,
                )
                ranked_ids = [item.node.id for item in result.ranked_nodes]
                metrics = retrieval_metrics(ranked_ids, example["relevant_node_ids"])
                metric_data = asdict(metrics)
                metric_data["f1"] = _f1(metrics.precision, metrics.recall)
                record = result.to_dict() | {
                    "condition": condition.name,
                    "example_id": example["id"],
                    "question_type": example["question_type"],
                    "metrics": metric_data,
                }
                if result.metadata["llm_used"] and pipeline.client is not None:
                    record["llm_call"] = dict(pipeline.client.last_call)
                records.append(record)
                stream.write(json.dumps(record) + "\n")

    summary = _summarize(records)
    by_type = _summarize_by_type(records)
    paired = _paired_bootstrap(
        records, samples=bootstrap_samples, seed=bootstrap_seed
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "by_question_type.json").write_text(
        json.dumps(by_type, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )
    llm_calls = [record["llm_call"] for record in records if "llm_call" in record]
    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "questions_path": str(source),
        "questions_sha256": _sha256(source),
        "question_count": len(questions),
        "conditions": [asdict(condition) for condition in conditions],
        "record_count": len(records),
        "answer_generation": False,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "model": pipeline.client.model if pipeline.client else None,
        "model_digest": FROZEN_MODEL_DIGEST if pipeline.client else None,
        "network_requests": sum(call["network_request"] for call in llm_calls),
        "cache_hits": sum(call["cache_hit"] for call in llm_calls),
        "prompt_tokens": sum(call["prompt_tokens"] or 0 for call in llm_calls),
        "completion_tokens": sum(
            call["completion_tokens"] or 0 for call in llm_calls
        ),
        "total_tokens": sum(call["total_tokens"] or 0 for call in llm_calls),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary
