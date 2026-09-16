"""Paired LLM-only versus graph-grounded answer generation and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient
from agp_research.planner import local_keyword_extraction


LLM_ONLY_SYSTEM = (
    "Answer the question concisely. Name the requested entities explicitly. "
    "If you do not know, say that the available information is insufficient."
)
GROUNDED_SYSTEM = (
    "Answer the question concisely using only the supplied graph context. "
    "Name the requested entities explicitly. If the context is insufficient, "
    "say so; do not add facts from outside the context."
)


@dataclass(frozen=True)
class AnswerMetrics:
    entity_precision: float
    entity_recall: float
    entity_f1: float
    mentioned_node_ids: list[str]
    unsupported_node_ids: list[str]
    context_faithfulness: float | None


def mentioned_node_ids(answer: str, graph: KnowledgeGraph) -> list[str]:
    """Resolve longest non-overlapping graph titles explicitly mentioned."""
    ids: list[str] = []
    for title in local_keyword_extraction(answer, graph):
        matched, _ = graph.exact_match([title])
        for node_id in matched:
            if node_id not in ids:
                ids.append(node_id)
    return ids


def answer_metrics(
    answer: str,
    relevant_node_ids: list[str],
    graph: KnowledgeGraph,
    *,
    context: str | None = None,
    excluded_node_ids: list[str] | None = None,
) -> AnswerMetrics:
    """Score explicit entity mentions and, when supplied, context faithfulness."""
    excluded = set(excluded_node_ids or [])
    predicted = [
        node_id for node_id in mentioned_node_ids(answer, graph)
        if node_id not in excluded
    ]
    expected = set(relevant_node_ids)
    hits = len(set(predicted) & expected)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if context is None:
        unsupported: list[str] = []
        faithfulness = None
    else:
        context_ids = set(mentioned_node_ids(context, graph))
        unsupported = [node_id for node_id in predicted if node_id not in context_ids]
        faithfulness = (
            (len(predicted) - len(unsupported)) / len(predicted)
            if predicted
            else 1.0
        )
    return AnswerMetrics(precision, recall, f1, predicted, unsupported, faithfulness)


def load_retrieval_contexts(
    path: str | Path, strategy: str
) -> dict[str, dict[str, Any]]:
    """Load exactly one saved retrieval record per example for a strategy."""
    selected: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("strategy") != strategy:
            continue
        example_id = record["example_id"]
        if example_id in selected:
            raise ValueError(f"Duplicate retrieval result for {example_id!r}")
        selected[example_id] = record
    return selected


def _blind_order(example_id: str, random_seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{random_seed}:{example_id}".encode()).digest()
    return ("llm_only", "agp_grounded") if digest[0] % 2 == 0 else ("agp_grounded", "llm_only")


def run_answer_quality_study(
    client: OpenAICompatibleClient,
    graph: KnowledgeGraph,
    questions_path: str | Path,
    retrieval_results_path: str | Path,
    output_dir: str | Path,
    *,
    retrieval_strategy: str,
    random_seed: int = 90055,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate paired answers from saved contexts and write auditable artifacts."""
    questions_file = Path(questions_path)
    retrieval_file = Path(retrieval_results_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    contexts = load_retrieval_contexts(retrieval_file, retrieval_strategy)
    missing = [question["id"] for question in questions if question["id"] not in contexts]
    if missing:
        raise ValueError(f"Retrieval results are missing examples: {missing}")

    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    answer_key: dict[str, dict[str, str]] = {}
    metric_totals = {
        "llm_only": {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0},
        "agp_grounded": {
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
            "context_faithfulness": 0.0,
        },
    }
    for question in questions:
        example_id = question["id"]
        context = contexts[example_id]["context"]
        llm_only = client.complete_text(LLM_ONLY_SYSTEM, f"Question: {question['question']}")
        llm_only_call = dict(client.last_call)
        grounded = client.complete_text(
            GROUNDED_SYSTEM,
            f"Question: {question['question']}\n\nGraph context:\n{context}",
        )
        grounded_call = dict(client.last_call)
        answers = {"llm_only": llm_only, "agp_grounded": grounded}
        metrics = {
            "llm_only": answer_metrics(
                llm_only, question["relevant_node_ids"], graph,
                excluded_node_ids=question.get("seed_node_ids"),
            ),
            "agp_grounded": answer_metrics(
                grounded, question["relevant_node_ids"], graph, context=context,
                excluded_node_ids=question.get("seed_node_ids"),
            ),
        }
        for condition, values in metrics.items():
            for name in metric_totals[condition]:
                metric_totals[condition][name] += getattr(values, name) or 0.0
        records.append(
            {
                "example_id": example_id,
                "question": question["question"],
                "question_type": question.get("question_type"),
                "relevant_node_ids": question["relevant_node_ids"],
                "reference_titles": [
                    graph.nodes[node_id].title for node_id in question["relevant_node_ids"]
                ],
                "retrieval_strategy": retrieval_strategy,
                "retrieval_context": context,
                "answers": answers,
                "metrics": {name: asdict(value) for name, value in metrics.items()},
                "llm_calls": {
                    "llm_only": llm_only_call,
                    "agp_grounded": grounded_call,
                },
            }
        )
        order = _blind_order(example_id, random_seed)
        pair_id = hashlib.sha256(f"pair:{random_seed}:{example_id}".encode()).hexdigest()[:12]
        blind_rows.append(
            {
                "pair_id": pair_id,
                "question": question["question"],
                "reference_titles": [
                    graph.nodes[node_id].title for node_id in question["relevant_node_ids"]
                ],
                "answer_a": answers[order[0]],
                "answer_b": answers[order[1]],
            }
        )
        answer_key[pair_id] = {"answer_a": order[0], "answer_b": order[1]}

    count = len(records)
    summary = {
        "questions": count,
        "model": client.model,
        "base_url": client.base_url,
        "retrieval_strategy": retrieval_strategy,
        "random_seed": random_seed,
        "automatic_metrics_are_conservative": True,
        "macro_average": {
            condition: {name: value / count for name, value in totals.items()}
            for condition, totals in metric_totals.items()
        },
        "input_sha256": {
            "questions": hashlib.sha256(questions_file.read_bytes()).hexdigest(),
            "retrieval_results": hashlib.sha256(retrieval_file.read_bytes()).hexdigest(),
        },
    }
    with (output / "answers.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "blind_review.jsonl").open("w", encoding="utf-8") as stream:
        for row in blind_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "answer_key.json").write_text(
        json.dumps(answer_key, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "human_ratings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "pair_id", "correctness_a_1_to_5", "correctness_b_1_to_5",
            "relevance_a_1_to_5", "relevance_b_1_to_5", "preferred_answer",
            "reviewer_notes",
        ])
        for row in blind_rows:
            writer.writerow([row["pair_id"], "", "", "", "", "", ""])
    return summary
