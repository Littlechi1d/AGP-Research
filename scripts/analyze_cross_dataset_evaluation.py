#!/usr/bin/env python3
"""Analyze frozen GitHub/Deezer outputs without making model or AGP calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ARMS = tuple(f"C{i}" for i in range(1, 9))
FIXED_ARMS = ("C2", "C3", "C4", "C5", "C6")
TYPES = ("direct", "comparison", "path", "similarity")
COLORS = {
    "github_musae": "#2563eb",
    "deezer_europe": "#ea580c",
}


def retrieval_f1(predicted: list[str], relevant: list[str]) -> float:
    predicted_set, relevant_set = set(predicted), set(relevant)
    overlap = len(predicted_set & relevant_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(relevant_set) if relevant_set else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def paired_bootstrap(
    left: list[float], right: list[float], *, samples: int = 10_000,
    seed: int = 90055,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must have the same nonzero length")
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    boot = sorted(
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    )
    return {
        "mean_difference": mean(differences),
        "ci95_low": boot[int(samples * 0.025)],
        "ci95_high": boot[int(samples * 0.975)],
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze_dataset(root: Path, name: str, spec: dict[str, str]) -> dict[str, Any]:
    questions_path = root / spec["questions"]
    contexts_path = root / spec["contexts"]
    answers_path = root / spec["answers"]
    context_manifest_path = contexts_path.parent / "manifest.json"
    answer_summary_path = answers_path.parent / "summary.json"
    context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    answer_summary = json.loads(answer_summary_path.read_text(encoding="utf-8"))
    if context_manifest["contexts_sha256"] != _hash(contexts_path):
        raise ValueError(f"{name} contexts fail their checksum")
    if answer_summary["answers_sha256"] != _hash(answers_path):
        raise ValueError(f"{name} answers fail their checksum")
    if context_manifest["conditions"] != list(("C0", *ARMS)):
        raise ValueError(f"{name} does not contain C0-C8")

    questions = {item["id"]: item for item in json.loads(questions_path.read_text())}
    contexts = _load_jsonl(contexts_path)
    if len(questions) != 40 or len(contexts) != 40:
        raise ValueError(f"{name} must contain 40 evaluation questions")

    per_question: dict[str, dict[str, float]] = {}
    by_type: dict[str, dict[str, list[float]]] = {
        kind: {arm: [] for arm in ARMS} for kind in TYPES
    }
    mapping = Counter()
    selections = Counter()
    for row in contexts:
        question = questions[row["example_id"]]
        intended = set(question["seed_node_ids"])
        matched = set(row["matched_seed_ids"])
        mapping["exact" if matched == intended else "partial" if matched else "empty"] += 1
        scores = {
            arm: retrieval_f1(
                row["conditions"][arm]["ranked_node_ids"],
                question["relevant_node_ids"],
            )
            for arm in ARMS
        }
        per_question[row["example_id"]] = scores
        for arm, value in scores.items():
            by_type[question["question_type"]][arm].append(value)
        selections[row["conditions"]["C8"]["reused_from"]] += 1

    vectors = {
        arm: [per_question[item_id][arm] for item_id in sorted(per_question)]
        for arm in ARMS
    }
    means = {arm: mean(values) for arm, values in vectors.items()}
    best_fixed = max(FIXED_ARMS, key=lambda arm: means[arm])
    comparisons = {
        f"{left}_minus_{right}": paired_bootstrap(vectors[left], vectors[right])
        for left, right in (("C1", best_fixed), ("C7", best_fixed), ("C8", best_fixed))
    }
    return {
        "question_count": len(contexts),
        "mapping_counts": dict(mapping),
        "retrieval_f1": means,
        "retrieval_f1_by_type": {
            kind: {arm: mean(values) for arm, values in arms.items()}
            for kind, arms in by_type.items()
        },
        "best_fixed_arm": best_fixed,
        "paired_bootstrap": comparisons,
        "answer_entity_f1": {
            arm: answer_summary["macro_average"][arm]["entity_f1"]
            for arm in ("C0", *ARMS)
        },
        "length_limited_answers": answer_summary["length_limited_answers"],
        "c8_selection_counts": dict(sorted(selections.items())),
        "input_sha256": {
            "questions": _hash(questions_path),
            "contexts": _hash(contexts_path),
            "answers": _hash(answers_path),
            "context_manifest": _hash(context_manifest_path),
            "answer_summary": _hash(answer_summary_path),
        },
    }


def _svg_bars(data: dict[str, dict[str, float]], title: str, ylabel: str) -> str:
    width, height = 980, 520
    left, top, plot_w, plot_h = 80, 60, 850, 370
    labels = list(next(iter(data.values())))
    datasets = list(data)
    maximum = max(value for values in data.values() for value in values.values()) * 1.14
    group_w = plot_w / len(labels)
    bar_w = min(32, group_w / (len(datasets) + 1))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.3}.label{font-size:13px}.value{font-size:11px}.title{font-size:20px;font-weight:600}</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{title}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * value / maximum
        parts += [
            f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="label">{value:.2f}</text>',
        ]
    parts += [
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" x2="{left+plot_w}" y1="{top+plot_h}" y2="{top+plot_h}" class="axis"/>',
        f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{ylabel}</text>',
    ]
    for i, label in enumerate(labels):
        center = left + group_w * (i + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="label">{label}</text>')
        for j, dataset in enumerate(datasets):
            value = data[dataset][label]
            x = center + (j - (len(datasets)-1)/2) * (bar_w + 5) - bar_w/2
            h = plot_h * value / maximum
            y = top + plot_h - h
            parts += [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{COLORS[dataset]}"/>',
                f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="value">{value:.3f}</text>',
            ]
    legend_x = left + plot_w - 250
    for j, dataset in enumerate(datasets):
        x = legend_x + j * 135
        parts += [
            f'<rect x="{x}" y="{height-35}" width="14" height="14" fill="{COLORS[dataset]}"/>',
            f'<text x="{x+20}" y="{height-23}" class="label">{"GitHub" if dataset == "github_musae" else "Deezer"}</text>',
        ]
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _svg_type_grid(results: dict[str, Any]) -> str:
    width, height = 1040, 600
    arms = list(ARMS)
    cell_w, cell_h = 95, 48
    left, top = 160, 105
    values = [
        results[dataset]["retrieval_f1_by_type"][kind][arm]
        for dataset in results for kind in TYPES for arm in arms
    ]
    maximum = max(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:600}.label{font-size:13px}.value{font-size:12px;font-weight:600}</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Retrieval F1 by question type</text>',
    ]
    for i, arm in enumerate(arms):
        parts.append(f'<text x="{left+i*cell_w+cell_w/2}" y="{top-12}" text-anchor="middle" class="label">{arm}</text>')
    row = 0
    for dataset in results:
        pretty = "GitHub MUSAE" if dataset == "github_musae" else "Deezer Europe"
        parts.append(f'<text x="12" y="{top+row*cell_h+18}" class="label">{pretty}</text>')
        for kind in TYPES:
            y = top + row * cell_h
            parts.append(f'<text x="145" y="{y+29}" text-anchor="end" class="label">{kind}</text>')
            for i, arm in enumerate(arms):
                value = results[dataset]["retrieval_f1_by_type"][kind][arm]
                opacity = 0.10 + 0.80 * value / maximum
                x = left + i * cell_w
                parts += [
                    f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" fill="#2563eb" fill-opacity="{opacity:.3f}"/>',
                    f'<text x="{x+(cell_w-4)/2}" y="{y+28}" text-anchor="middle" class="value">{value:.3f}</text>',
                ]
            row += 1
        row += 0.6
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _report(results: dict[str, Any]) -> str:
    lines = [
        "# Frozen evaluation statistical analysis",
        "",
        "This analysis uses only saved C0–C8 outputs. It makes no LLM or AGP calls.",
        "Paired 95% confidence intervals use 10,000 bootstrap resamples with seed 90055.",
        "",
        "## Paired retrieval-F1 comparisons",
        "",
        "| Dataset | Comparison | Mean difference | 95% bootstrap CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset, result in results.items():
        pretty = "GitHub MUSAE" if dataset == "github_musae" else "Deezer Europe"
        best = result["best_fixed_arm"]
        for left in ("C1", "C7", "C8"):
            item = result["paired_bootstrap"][f"{left}_minus_{best}"]
            lines.append(
                f'| {pretty} | {left} − {best} | {item["mean_difference"]:+.4f} | '
                f'[{item["ci95_low"]:+.4f}, {item["ci95_high"]:+.4f}] |'
            )
    lines += [
        "",
        "Intervals crossing zero do not establish a clear paired difference at the",
        "95% bootstrap level. These intervals are descriptive and are not corrected",
        "for the multiple arm comparisons in the broader study.",
        "",
        "## Figures",
        "",
        "- `retrieval_f1_by_condition.svg`: overall retrieval comparison.",
        "- `retrieval_f1_by_question_type.svg`: question-type heatmap.",
        "- `answer_entity_f1_by_condition.svg`: conservative answer entity coverage.",
        "",
        "## Interpretation",
        "",
        "Direct neighbours lead overall because one quarter of the balanced questions",
        "ask directly for neighbours. They score zero on similarity questions, where",
        "propagation is necessary. C7 equals C2 because it always used its fallback.",
        "C8 does not beat the strongest fixed arm on either dataset. The figures",
        "therefore support question-type heterogeneity, but not successful adaptation",
        "by the two frozen selectors.",
        "",
    ]
    return "\n".join(lines)


def analyze(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    specs = {
        "github_musae": {
            "questions": "data/github_musae/benchmark/questions_eval.json",
            "contexts": "results/github_frozen_eval_contexts_c0_c8_20260929/contexts.jsonl",
            "answers": "results/github_frozen_eval_answers_c0_c8_20260929/answers.jsonl",
        },
        "deezer_europe": {
            "questions": "data/deezer_europe/benchmark/questions_eval.json",
            "contexts": "results/deezer_frozen_eval_contexts_c0_c8_20260929/contexts.jsonl",
            "answers": "results/deezer_frozen_eval_answers_c0_c8_20260929/answers.jsonl",
        },
    }
    results = {name: analyze_dataset(root, name, spec) for name, spec in specs.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-analysis-", dir=output.parent) as temporary:
        staged = Path(temporary)
        (staged / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        (staged / "REPORT.md").write_text(_report(results), encoding="utf-8")
        with (staged / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["dataset", "condition", "retrieval_f1", "answer_entity_f1"])
            for dataset, result in results.items():
                for arm in ARMS:
                    writer.writerow([dataset, arm, result["retrieval_f1"][arm], result["answer_entity_f1"][arm]])
        retrieval = {dataset: result["retrieval_f1"] for dataset, result in results.items()}
        answers = {dataset: result["answer_entity_f1"] for dataset, result in results.items()}
        (staged / "retrieval_f1_by_condition.svg").write_text(
            _svg_bars(retrieval, "Frozen retrieval performance", "Mean per-question retrieval F1")
        )
        (staged / "answer_entity_f1_by_condition.svg").write_text(
            _svg_bars(answers, "Conservative answer entity coverage", "Mean answer entity F1")
        )
        (staged / "retrieval_f1_by_question_type.svg").write_text(_svg_type_grid(results))
        staged.rename(output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = analyze(args.root.resolve(), args.output.resolve())
    print(json.dumps({
        dataset: {
            "best_fixed_arm": result["best_fixed_arm"],
            "paired_bootstrap": result["paired_bootstrap"],
        }
        for dataset, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
