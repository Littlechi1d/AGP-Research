#!/usr/bin/env python3
"""Tune approximate entity matching on labelled development questions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.graph import KnowledgeGraph
from agp_research.mapping_evaluation import mapping_metrics, select_configuration


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--keyword-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.70, .75, .80, .85, .90, .95])
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, .03, .05, .10])
    args = parser.parse_args()

    graph = KnowledgeGraph.from_csv(args.nodes, args.edges)
    examples = json.loads(args.questions.read_text(encoding="utf-8"))
    keyword_rows = [
        json.loads(line)
        for line in args.keyword_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keywords_by_id = {row["example_id"]: row["keywords"] for row in keyword_rows}
    missing = sorted({example["id"] for example in examples} - set(keywords_by_id))
    if missing:
        raise ValueError(f"Keyword results are missing examples: {missing}")

    exact = mapping_metrics(graph, examples, keywords_by_id, mode="exact")
    grid = [
        mapping_metrics(
            graph, examples, keywords_by_id, mode="approximate",
            threshold=threshold, margin=margin,
        )
        for threshold in args.thresholds
        for margin in args.margins
    ]
    selected = select_configuration(grid)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "grid_results.json").write_text(
        json.dumps([_compact(item) for item in grid], indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "selected_details.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "selection_rule": "maximize precision, then F1, recall, threshold, and margin",
        "inputs": {
            str(path): _hash(path)
            for path in (args.nodes, args.edges, args.questions, args.keyword_results)
        },
        "thresholds": args.thresholds,
        "margins": args.margins,
        "exact_baseline": _compact(exact),
        "selected": _compact(selected),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Approximate entity-mapping development result

This tuning used only the labelled development questions. It did not inspect or
overwrite the frozen held-out results.

## Selection rule

Maximize micro precision first because a false seed sends graph propagation into
the wrong region; break ties by F1, recall, threshold, and ambiguity margin.

## Result

- Exact baseline: precision {exact['precision']:.3f}, recall {exact['recall']:.3f}, F1 {exact['f1']:.3f}, exact sets {exact['exact_sets']}/{exact['examples']}.
- Selected approximate configuration: threshold `{selected['threshold']}`, margin `{selected['margin']}`.
- Approximate mapping: precision {selected['precision']:.3f}, recall {selected['recall']:.3f}, F1 {selected['f1']:.3f}, exact sets {selected['exact_sets']}/{selected['examples']}.

The compound recovery pass joins adjacent unresolved LLM keywords before
individual fuzzy matching. This recovers long titles that the LLM split into
several pieces.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest["selected"], indent=2))


if __name__ == "__main__":
    main()
