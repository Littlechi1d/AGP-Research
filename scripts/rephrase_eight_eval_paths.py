#!/usr/bin/env python3
"""Create a new evaluation candidate with plainer path-question wording."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_facebook_questions import (  # noqa: E402
    read_graph,
    unique_three_hop_path,
    valid_question,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain_path_question(start: str, end: str) -> str:
    return (
        f"To get from {start} to {end} using the fewest page-to-page connections, "
        "which two pages do you pass through, in order?"
    )


def rephrase_path_candidate(
    previous_dir: str | Path,
    nodes_path: str | Path,
    edges_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Preserve IDs and answer labels while changing only path question text."""
    previous, output = Path(previous_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    old_questions = previous / "questions.json"
    old_review = previous / "criteria_review.csv"
    old_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest["questions_sha256"] != _hash(old_questions):
        raise ValueError("previous questions do not match their manifest")
    if old_manifest["criteria_review_sha256"] != _hash(old_review):
        raise ValueError("previous review form does not match its manifest")
    nodes, edges = Path(nodes_path), Path(edges_path)
    if old_manifest["source_sha256"]["nodes"] != _hash(nodes):
        raise ValueError("node file does not match previous candidate")
    if old_manifest["source_sha256"]["edges"] != _hash(edges):
        raise ValueError("edge file does not match previous candidate")
    titles, adjacency = read_graph(nodes, edges)
    questions = json.loads(old_questions.read_text(encoding="utf-8"))
    with old_review.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        review_rows = list(reader)
    if fieldnames is None or len(review_rows) != len(questions):
        raise ValueError("review form does not match question count")
    if [row["example_id"] for row in review_rows] != [q["id"] for q in questions]:
        raise ValueError("review rows are not aligned with questions")
    if any(row.get("reviewer_id") or row.get("notes") or
           row.get("question_unambiguous_yes_no") or
           row.get("labels_correct_yes_no") for row in review_rows):
        raise ValueError("previous review form is no longer blank")

    changed_ids = []
    for question, review in zip(questions, review_rows):
        if question["question_type"] != "path":
            continue
        start_id, end_id = question["seed_node_ids"]
        expected = unique_three_hop_path(start_id, end_id, adjacency)
        if expected is None or list(expected) != question["relevant_node_ids"]:
            raise ValueError(f"path answer order is invalid for {question['id']}")
        revised = plain_path_question(titles[start_id], titles[end_id])
        if not valid_question(revised, [start_id, end_id], titles):
            raise ValueError(f"new wording changes seed extraction for {question['id']}")
        if revised == question["question"]:
            raise ValueError(f"path question was already rephrased: {question['id']}")
        question["question"] = review["question"] = revised
        changed_ids.append(question["id"])
    if not changed_ids:
        raise ValueError("candidate contains no path questions")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-rephrase-paths-", dir=output.parent) as temp:
        staged = Path(temp)
        questions_file = staged / "questions.json"
        questions_file.write_text(
            json.dumps(questions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        review_file = staged / "criteria_review.csv"
        with review_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(review_rows)
        manifest = {
            **old_manifest,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "candidate_pending_independent_criteria_review",
            "revision_from": str(previous),
            "revision_changes": [
                "Rephrased path questions in plain language while preserving shortest-path meaning and answer order."
            ],
            "previous_questions_sha256": old_manifest["questions_sha256"],
            "previous_criteria_review_sha256": old_manifest["criteria_review_sha256"],
            "rephrased_path_examples": changed_ids,
            "questions_sha256": _hash(questions_file),
            "criteria_review_sha256": _hash(review_file),
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staged / "REPORT.md").write_text(
            "# Eight-condition evaluation candidate: plainer path wording\n\n"
            "This unscored candidate supersedes V4. Only the path question wording changed: "
            "each now asks which two pages you pass through using the fewest "
            "page-to-page connections. All question IDs, seed IDs, provisional "
            "answer IDs, and answer order are unchanged. The previous candidate "
            "and input hashes are recorded in `manifest.json`.\n\n"
            "This set is **not frozen or independently reviewed**. A reviewer must "
            "check every question and complete a copy of `criteria_review.csv` "
            "before the one-time eight-arm evaluation.\n",
            encoding="utf-8",
        )
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rephrase_path_candidate(
        args.previous, args.nodes, args.edges, args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
