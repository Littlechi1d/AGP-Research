"""Development-only grid search using the unchanged experiment runner.

Run from the project root with `python3 scripts/tune_fixed.py --output-dir ...`.
The default grid tests 36 configurations and never reads the test questions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.evaluation import run_experiment
from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters
from agp_research.pipeline import AGPPipeline
from agp_research.planner import local_keyword_extraction


def f1(precision: float, recall: float) -> float:
    """F1 for ONE question; the search averages these per-question values."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def selection_key(row: dict) -> tuple:
    """Predeclared order: F1, recall, smaller k, smaller depth, smaller decay."""
    p = row["parameters"]
    return (-row["metrics"]["f1"], -row["metrics"]["recall"],
            p["top_k"], p["depth"], p["decay"])


def summarize(path: Path, questions: dict) -> dict:
    grouped = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        metrics = dict(row["metrics"])
        metrics["f1"] = f1(metrics["precision"], metrics["recall"])
        metrics["latency"] = row["elapsed_seconds"]
        grouped["overall"].append(metrics)
        grouped[questions[row["example_id"]]["question_type"]].append(metrics)
    if len(grouped["overall"]) != len(questions):
        raise ValueError("Incomplete experiment output")
    return {
        group: {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        for group, rows in grouped.items()
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=Path("data/facebook_large/nodes.csv"))
    parser.add_argument("--edges", type=Path, default=Path("data/facebook_large/edges.csv"))
    parser.add_argument("--questions", type=Path,
                        default=Path("data/facebook_large/facebook_questions_dev.json"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; refuses to overwrite an existing run")
    args = parser.parse_args()
    if "test" in args.questions.stem.lower():
        parser.error("Use development questions, not the held-out test set")
    graph = KnowledgeGraph.from_csv(args.nodes, args.edges)
    examples = json.loads(args.questions.read_text(encoding="utf-8"))
    if not examples or len({q["id"] for q in examples}) != len(examples):
        parser.error("Questions must be nonempty with unique IDs")
    for q in examples:
        if not q["relevant_node_ids"] or not set(q["relevant_node_ids"]) <= graph.nodes.keys():
            parser.error(f"Invalid labels: {q['id']}")
        matched, unmatched = graph.exact_match(local_keyword_extraction(q["question"], graph))
        if unmatched or set(matched) != set(q["seed_node_ids"]):
            parser.error(f"Unexpected seed extraction: {q['id']}")
    questions = {q["id"]: q for q in examples}
    parameters = [AGPParameters(d, a, k).validate()
                  for d, a, k in itertools.product((1, 2, 3), (0.3, 0.5, 0.7, 0.9), (5, 10, 20))]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    project = Path(__import__("agp_research").__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                              capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=project,
                            capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "git_revision": revision, "git_status": status,
        "source_sha256": {p.name: digest(p) for p in sorted((project / "agp_research").glob("*.py"))},
        "tuner_sha256": digest(Path(__file__)),
        "input_sha256": {str(p.resolve()): digest(p) for p in (args.nodes, args.edges, args.questions)},
        "question_count": len(examples), "backend": "python-exact", "llm_calls": 0,
        "grid": [p.__dict__ for p in parameters],
        "objective": "Mean of per-question F1 (not F1 of macro precision/recall)",
        "tie_breaks": ["higher macro recall", "smaller top_k", "smaller depth", "smaller decay"],
        "precision_denominator": "actual number of unique nodes returned, not a padded k",
        "held_out_test_evaluated": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    pipeline = AGPPipeline(graph, None)  # No credentials are loaded; no API calls.
    rows = []
    for number, params in enumerate(parameters, 1):
        name = f"fixed_d{params.depth}_a{str(params.decay).replace('.', '')}_k{params.top_k}"
        path = args.output_dir / f"{name}.jsonl"
        run_experiment(pipeline, args.questions, path, ["fixed"], fixed_parameters=params)
        stats = summarize(path, questions)
        rows.append({"name": name, "parameters": params.__dict__, "metrics": stats.pop("overall"),
                     "by_type": stats, "trace": path.name})
        write_json(args.output_dir / "grid_results.json", rows)
        print(f"[{number:02d}/36] {name}: F1={rows[-1]['metrics']['f1']:.6f}, "
              f"recall={rows[-1]['metrics']['recall']:.6f}", flush=True)

    comparisons = {}
    for strategy in ("fixed", "rules"):
        path = args.output_dir / f"reference_{strategy}.jsonl"
        run_experiment(pipeline, args.questions, path, [strategy])
        comparisons[strategy] = summarize(path, questions)
        print(f"Reference {strategy} completed", flush=True)
    ranked = sorted(rows, key=selection_key)
    best_at_k = {str(k): min((r for r in rows if r["parameters"]["top_k"] == k), key=selection_key)
                 for k in (5, 10, 20)}
    selection = {"selected": ranked[0], "best_at_each_k": best_at_k,
                 "reference_results": comparisons,
                 "scope": "Development-selected best within the declared grid; not a global optimum",
                 "pipeline_runs": len(examples) * (len(parameters) + 2)}
    write_json(args.output_dir / "selection.json", selection)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "manifest.json", manifest)
    print("Selected: " + json.dumps(ranked[0]["parameters"]), flush=True)


if __name__ == "__main__":
    main()
