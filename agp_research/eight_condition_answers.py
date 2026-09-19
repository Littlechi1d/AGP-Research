"""Generate and blind eight answers from already saved retrieval contexts."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agp_research.answer_evaluation import answer_metrics
from agp_research.eight_condition_contexts import CONDITION_IDS
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient


ANSWER_SYSTEM = (
    "Answer the question concisely. Name requested entities explicitly. "
    "If graph context is supplied, use only that evidence for graph facts. "
    "If no context is supplied, answer from your own knowledge. "
    "If the available information is insufficient, say so explicitly."
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order(example_id: str, random_seed: int, values: list[str]) -> list[str]:
    digest = hashlib.sha256(f"{random_seed}:{example_id}".encode()).digest()
    result = values.copy()
    random.Random(int.from_bytes(digest, "big")).shuffle(result)
    return result


def load_eight_contexts(path: str | Path) -> dict[str, dict[str, Any]]:
    """Validate eight-condition rows and reject duplicate question IDs."""
    contexts: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row["example_id"]
        if example_id in contexts:
            raise ValueError(f"duplicate context row for {example_id}")
        conditions = row["conditions"]
        if tuple(conditions) != CONDITION_IDS:
            raise ValueError(f"incomplete or reordered conditions for {example_id}")
        selected = conditions["C7"]["reused_from"]
        if selected not in CONDITION_IDS[2:7]:
            raise ValueError(f"C7 must select one of C2-C6 for {example_id}")
        if conditions["C0"]["context"]:
            raise ValueError(f"C0 must have no graph context for {example_id}")
        if (
            conditions["C7"]["context"] != conditions[selected]["context"]
            or conditions["C7"]["ranked_node_ids"]
            != conditions[selected]["ranked_node_ids"]
        ):
            raise ValueError(f"C7 must reuse its chosen arm exactly for {example_id}")
        contexts[example_id] = row
    return contexts


def run_eight_condition_answers(
    client: OpenAICompatibleClient,
    graph: KnowledgeGraph,
    questions_path: str | Path,
    contexts_path: str | Path,
    output_dir: str | Path,
    *,
    random_seed: int = 90055,
    limit: int | None = None,
    input_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Make seven model calls per question; C7 copies its selected fixed answer."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    questions_file = Path(questions_path)
    contexts_file = Path(contexts_path)
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    question_ids = [question["id"] for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("question IDs must be unique")
    context_manifest_path = contexts_file.parent / "manifest.json"
    if not context_manifest_path.is_file():
        raise ValueError("context manifest is missing")
    context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    if context_manifest.get("contexts_sha256") != _hash(contexts_file):
        raise ValueError("saved contexts do not match their manifest checksum")
    context_inputs = context_manifest.get("input_sha256", {})
    if context_inputs.get("questions") != _hash(questions_file):
        raise ValueError("questions do not match the saved context manifest")
    for name in ("nodes", "edges"):
        if input_paths and name in input_paths and name in context_inputs:
            if context_inputs[name] != _hash(Path(input_paths[name])):
                raise ValueError(f"{name} do not match the saved context manifest")
    contexts = load_eight_contexts(contexts_file)
    missing = sorted(set(question_ids) - set(contexts))
    if missing:
        raise ValueError(f"contexts missing questions: {missing}")
    for question in questions:
        if contexts[question["id"]]["question"] != question["question"]:
            raise ValueError(f"question differs from saved context: {question['id']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        condition: {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0}
        for condition in CONDITION_IDS
    }
    faithfulness_totals = {condition: 0.0 for condition in CONDITION_IDS[1:]}
    prompt_tokens = {condition: 0 for condition in CONDITION_IDS[:7]}
    completion_tokens = {condition: 0 for condition in CONDITION_IDS[:7]}
    request_count = cache_hits = 0
    with tempfile.TemporaryDirectory(prefix="agp-eight-answers-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        key: dict[str, dict[str, str]] = {}
        blind_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []
        with (temporary_path / "answers.jsonl").open("w", encoding="utf-8") as stream:
            for question in questions:
                example_id = question["id"]
                context_row = contexts[example_id]
                answers: dict[str, str] = {}
                calls: dict[str, dict[str, Any]] = {}
                generation_order = _order(
                    example_id, random_seed, list(CONDITION_IDS[:7])
                )
                for condition in generation_order:
                    graph_context = context_row["conditions"][condition]["context"]
                    user = f"Question: {question['question']}"
                    if condition != "C0":
                        user += f"\n\nGraph context:\n{graph_context}"
                    answers[condition] = client.complete_text(ANSWER_SYSTEM, user)
                    calls[condition] = dict(client.last_call)
                    request_count += int(bool(calls[condition].get("network_request", True)))
                    cache_hits += int(bool(calls[condition].get("cache_hit", False)))
                    prompt_tokens[condition] += calls[condition].get("prompt_tokens") or 0
                    completion_tokens[condition] += calls[condition].get("completion_tokens") or 0

                selected = context_row["conditions"]["C7"]["reused_from"]
                answers["C7"] = answers[selected]
                calls["C7"] = {"copied_from": selected, "new_model_call": False}
                answers = {condition: answers[condition] for condition in CONDITION_IDS}
                calls = {condition: calls[condition] for condition in CONDITION_IDS}
                metrics = {}
                for condition in CONDITION_IDS:
                    graph_context = (
                        None if condition == "C0"
                        else context_row["conditions"][condition]["context"]
                    )
                    metric = answer_metrics(
                        answers[condition], question["relevant_node_ids"], graph,
                        context=graph_context,
                        excluded_node_ids=context_row["matched_seed_ids"],
                    )
                    metrics[condition] = asdict(metric)
                    for name in totals[condition]:
                        totals[condition][name] += getattr(metric, name)
                    if condition != "C0":
                        faithfulness_totals[condition] += metric.context_faithfulness or 0.0

                reference_titles = [
                    graph.nodes[node_id].title
                    for node_id in question["relevant_node_ids"]
                ]
                stream.write(json.dumps({
                    "example_id": example_id,
                    "question": question["question"],
                    "question_type": question.get("question_type"),
                    "reference_node_ids": question["relevant_node_ids"],
                    "reference_titles": reference_titles,
                    "matched_seed_ids": context_row["matched_seed_ids"],
                    "generation_order": generation_order,
                    "answers": answers,
                    "automatic_metrics": metrics,
                    "llm_calls": calls,
                    "context_characters": {
                        condition: context_row["conditions"][condition]["context_characters"]
                        for condition in CONDITION_IDS
                    },
                    "adaptive_reused_from": selected,
                }, ensure_ascii=False) + "\n")

                labels = "ABCDEFGH"
                randomized = _order(example_id, random_seed + 1, list(CONDITION_IDS))
                mapping = dict(zip(labels, randomized))
                key[example_id] = mapping
                blind_rows.append({
                    "example_id": example_id,
                    "question": question["question"],
                    "reference_titles": reference_titles,
                    "answers": {label: answers[condition] for label, condition in mapping.items()},
                })
                support_rows.append({
                    "example_id": example_id,
                    "question": question["question"],
                    "answers_with_context": {
                        label: {
                            "answer": answers[condition],
                            "context": context_row["conditions"][condition]["context"],
                        }
                        for label, condition in mapping.items() if condition != "C0"
                    },
                })

        for name, rows in (
            ("blind_review.jsonl", blind_rows),
            ("faithfulness_review.jsonl", support_rows),
        ):
            (temporary_path / name).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
        (temporary_path / "answer_key.json").write_text(
            json.dumps(key, indent=2) + "\n", encoding="utf-8"
        )
        with (temporary_path / "human_ratings.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([
                "reviewer_id", "example_id", "blind_label",
                "correctness_1_to_5", "completeness_1_to_5",
                "abstention_appropriate_yes_no_na", "notes",
            ])
            for row in blind_rows:
                for label in "ABCDEFGH":
                    writer.writerow(["", row["example_id"], label, "", "", "", ""])
        with (temporary_path / "faithfulness_ratings.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([
                "reviewer_id", "example_id", "blind_label",
                "factual_support_1_to_5", "unsupported_claims_count", "notes",
            ])
            for row in support_rows:
                for label in row["answers_with_context"]:
                    writer.writerow(["", row["example_id"], label, "", "", ""])

        count = len(questions)
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "question_count": count,
            "conditions": list(CONDITION_IDS),
            "model": client.model,
            "base_url": client.base_url,
            "temperature": 0,
            "random_seed": random_seed,
            "answer_system_prompt_sha256": hashlib.sha256(ANSWER_SYSTEM.encode()).hexdigest(),
            "generation_calls_per_question": 7,
            "adaptive_answer_reuse": True,
            "network_requests": request_count,
            "cache_hits": cache_hits,
            "prompt_tokens_by_generated_condition": prompt_tokens,
            "completion_tokens_by_generated_condition": completion_tokens,
            "macro_average": {
                condition: {
                    **{name: value / count if count else 0.0 for name, value in totals[condition].items()},
                    "context_faithfulness": (
                        faithfulness_totals[condition] / count if count else 0.0
                    ) if condition != "C0" else None,
                }
                for condition in CONDITION_IDS
            },
            "automatic_metrics_are_conservative": True,
            "input_sha256": {
                "questions": _hash(questions_file),
                "contexts": _hash(contexts_file),
                "context_manifest": _hash(context_manifest_path),
                **{
                    name: _hash(Path(path)) for name, path in (input_paths or {}).items()
                },
            },
            "answers_sha256": _hash(temporary_path / "answers.jsonl"),
        }
        (temporary_path / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(output)
    return summary
