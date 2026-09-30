#!/usr/bin/env python3
"""Run a small blinded semantic review over frozen saved answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from agp_research.llm import OpenAICompatibleClient


CONDITIONS = ("C0", "C1", "C2", "C8")
QUESTION_TYPES = ("direct", "comparison", "path", "similarity")
RUBRIC = """You are a blinded evaluator of an answer to a graph question.
You receive the question, the verified complete gold-answer entity-title set, and one candidate answer.
Do not infer or discuss which retrieval method produced the answer.

IMPORTANT: Every entity in the gold-answer set is already verified to satisfy the
relation requested by the question. The set is the correct answer, not a list of
context entities. It normally excludes seed entities named in the question and
intermediate entities not requested as the answer. Do not demand independent edge
evidence and do not penalize the absence of seed entities from the gold-answer set.
Judge whether the candidate's final answer names the gold entities. A response that
claims insufficient information or says no entity exists when the gold set is nonempty
must receive correctness 1 and completeness 1.

Score correctness from 1 to 5:
5 = the conclusion is correct, contains no material false graph claim, and answers the question directly.
4 = essentially correct with only a minor imprecision or harmless extra statement.
3 = partly correct but has a material omission, ambiguity, or unsupported conclusion.
2 = contains little correct answer content and major errors or contradictions.
1 = incorrect, says no answer when reference entities exist, or provides no usable answer.

Score completeness from 1 to 5:
5 = explicitly gives every reference entity in the requested form/order where relevant.
4 = gives most reference entities, with a small omission.
3 = gives roughly half or an incomplete final answer.
2 = gives only a small part of the reference answer.
1 = gives none of the reference answer.

Judge only against the supplied gold-answer entities and the requested relation. Return JSON
with integer keys "correctness" and "completeness" and a concise string key "reason"."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _order(key: str, values: tuple[str, ...], seed: int) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    result = list(values)
    random.Random(int.from_bytes(digest, "big")).shuffle(result)
    return result


def select_question_ids(questions: list[dict[str, Any]], seed: int = 90055) -> list[str]:
    """Select one question per type without using answers, contexts, or scores."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in questions:
        grouped[item["question_type"]].append(item["id"])
    if set(grouped) != set(QUESTION_TYPES):
        raise ValueError("questions must contain the four expected types")
    selected = []
    for kind in QUESTION_TYPES:
        candidates = sorted(grouped[kind])
        selected.append(min(
            candidates,
            key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
        ))
    return selected


def _validate_score(result: dict[str, Any]) -> dict[str, Any]:
    for name in ("correctness", "completeness"):
        value = result.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ValueError(f"judge returned invalid {name}")
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("judge returned an invalid reason")
    return {
        "correctness": result["correctness"],
        "completeness": result["completeness"],
        "reason": reason.strip(),
    }


def run_pilot(
    client: OpenAICompatibleClient,
    root: Path,
    output: Path,
    *,
    random_seed: int = 90055,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    specs = {
        "github_musae": {
            "questions": root / "data/github_musae/benchmark/questions_eval.json",
            "answers": root / "results/github_frozen_eval_answers_c0_c8_20260929/answers.jsonl",
        },
        "deezer_europe": {
            "questions": root / "data/deezer_europe/benchmark/questions_eval.json",
            "answers": root / "results/deezer_frozen_eval_answers_c0_c8_20260929/answers.jsonl",
        },
    }

    # Select IDs before opening answer files, so answer quality cannot influence sampling.
    questions_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    selected_by_dataset: dict[str, list[str]] = {}
    for dataset, spec in specs.items():
        questions = json.loads(spec["questions"].read_text(encoding="utf-8"))
        questions_by_dataset[dataset] = {item["id"]: item for item in questions}
        selected_by_dataset[dataset] = select_question_ids(questions, random_seed)

    blind_cases = []
    answer_key: dict[str, dict[str, str]] = {}
    for dataset, spec in specs.items():
        answers = {row["example_id"]: row for row in _rows(spec["answers"])}
        for example_id in selected_by_dataset[dataset]:
            question = questions_by_dataset[dataset][example_id]
            answer_row = answers[example_id]
            order = _order(f"{dataset}:{example_id}", CONDITIONS, random_seed)
            mapping = dict(zip("ABCD", order))
            answer_key[example_id] = mapping
            blind_cases.append({
                "dataset": dataset,
                "example_id": example_id,
                "question_type": question["question_type"],
                "question": question["question"],
                "reference_titles": answer_row["reference_titles"],
                "answers": {label: answer_row["answers"][condition] for label, condition in mapping.items()},
            })

    judgments = []
    request_count = cache_hits = 0
    # Complete and save all blind judgments before revealing the answer key below.
    for case in blind_cases:
        for label in "ABCD":
            user = (
                f"Question: {case['question']}\n\n"
                f"Verified gold-answer entities: {json.dumps(case['reference_titles'], ensure_ascii=False)}\n\n"
                f"Candidate answer {label}:\n{case['answers'][label]}"
            )
            rating = _validate_score(client.complete_json(RUBRIC, user))
            call = dict(client.last_call)
            request_count += int(bool(call.get("network_request", True)))
            cache_hits += int(bool(call.get("cache_hit", False)))
            judgments.append({
                "dataset": case["dataset"],
                "example_id": case["example_id"],
                "question_type": case["question_type"],
                "blind_label": label,
                **rating,
                "judge_call": call,
            })
            print(f"{case['example_id']} {label}: {rating['correctness']}/{rating['completeness']}", flush=True)

    revealed = []
    totals: dict[str, dict[str, list[float]]] = {
        condition: {"correctness": [], "completeness": []} for condition in CONDITIONS
    }
    by_dataset: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {
            condition: {"correctness": [], "completeness": []}
            for condition in CONDITIONS
        }
    )
    for item in judgments:
        condition = answer_key[item["example_id"]][item["blind_label"]]
        row = {**item, "condition": condition}
        revealed.append(row)
        for metric in ("correctness", "completeness"):
            totals[condition][metric].append(item[metric])
            by_dataset[item["dataset"]][condition][metric].append(item[metric])

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_only": True,
        "model_assisted_review": True,
        "same_model_family_as_answer_generator": client.model,
        "independent_human_review": False,
        "random_seed": random_seed,
        "sample_rule": "one deterministic hash-selected question per type and dataset",
        "question_count": len(blind_cases),
        "conditions": list(CONDITIONS),
        "rating_count": len(revealed),
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "selection_completed_before_answer_access": True,
        "all_judgments_completed_before_key_reveal": True,
        "network_requests": request_count,
        "cache_hits": cache_hits,
        "mean_scores": {
            condition: {metric: mean(values) for metric, values in metrics.items()}
            for condition, metrics in totals.items()
        },
        "mean_scores_by_dataset": {
            dataset: {
                condition: {metric: mean(values) for metric, values in metrics.items()}
                for condition, metrics in conditions.items()
            }
            for dataset, conditions in by_dataset.items()
        },
        "selected_question_ids": selected_by_dataset,
        "input_sha256": {
            f"{dataset}_{name}": _hash(path)
            for dataset, spec in specs.items() for name, path in spec.items()
        },
    }

    lines = [
        "# Small blinded semantic-review pilot",
        "",
        "This is an exploratory model-assisted review of saved frozen answers. It",
        "does not rerun retrieval or answer generation. One question per type and",
        "dataset was selected by a deterministic hash before answer files were opened.",
        "",
        "## Mean blind-judge scores",
        "",
        "| Condition | Correctness (1–5) | Completeness (1–5) |",
        "| --- | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        values = summary["mean_scores"][condition]
        lines.append(f'| {condition} | {values["correctness"]:.3f} | {values["completeness"]:.3f} |')
    lines += [
        "",
        "## Scores by dataset",
        "",
        "| Dataset | Condition | Correctness | Completeness |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset, conditions in summary["mean_scores_by_dataset"].items():
        pretty = "GitHub MUSAE" if dataset == "github_musae" else "Deezer Europe"
        for condition in CONDITIONS:
            values = conditions[condition]
            lines.append(
                f'| {pretty} | {condition} | {values["correctness"]:.3f} | '
                f'{values["completeness"]:.3f} |'
            )
    lines += [
        "",
        "C2 has the highest mean score in this eight-question pilot. C0 receives",
        "1/5 because it abstains despite nonempty verified gold-answer sets. C1 and",
        "C8 vary by question depending on whether the saved answer explicitly reaches",
        "the requested entity list before truncation.",
        "",
        "## Interpretation limits",
        "",
        "The judge saw only randomized A–D labels, the question, the reference entity",
        "set, and one answer at a time. The condition key was revealed only after all",
        "32 ratings were complete. However, the judge is the same local model used for",
        "answer generation, the sample contains only eight questions, and the reference",
        "criteria remain topology-defined entity sets. These ratings are a pilot and",
        "must not be presented as independent human evidence or definitive semantic",
        "quality estimates.",
        "",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-blind-pilot-", dir=output.parent) as temporary:
        staged = Path(temporary)
        for name, rows in (
            ("blind_cases.jsonl", blind_cases),
            ("judgments_blind.jsonl", judgments),
            ("ratings_revealed.jsonl", revealed),
        ):
            (staged / name).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
        (staged / "answer_key.json").write_text(json.dumps(answer_key, indent=2) + "\n")
        (staged / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (staged / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        staged.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--random-seed", type=int, default=90055)
    args = parser.parse_args()
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("configure the local LLM in .env")
    print(json.dumps(run_pilot(
        client, args.root.resolve(), args.output.resolve(), random_seed=args.random_seed
    ), indent=2))


if __name__ == "__main__":
    main()
