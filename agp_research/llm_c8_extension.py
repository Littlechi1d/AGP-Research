"""Add an LLM-selected C8 arm to a completed C0-C7 experiment."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agp_research.eight_condition_contexts import CONDITION_IDS
from agp_research.llm_pair_selector import LLMPairSelector, SYSTEM_PROMPT


NINE_CONDITION_IDS = (*CONDITION_IDS, "C8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def add_llm_c8(
    selector: LLMPairSelector,
    source_contexts: str | Path,
    source_answers: str | Path,
    context_output: str | Path,
    answer_output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select C8 before reading answer metrics, then copy the chosen fixed arm."""
    contexts_path = Path(source_contexts)
    answers_path = Path(source_answers)
    context_dir = Path(context_output)
    answer_dir = Path(answer_output)
    if context_dir.exists() or answer_dir.exists():
        raise FileExistsError("refusing to overwrite C8 output")

    source_context_manifest = json.loads(
        (contexts_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    source_answer_summary = json.loads(
        (answers_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    if source_context_manifest.get("contexts_sha256") != _hash(contexts_path):
        raise ValueError("source contexts fail their checksum")
    if source_answer_summary.get("answers_sha256") != _hash(answers_path):
        raise ValueError("source answers fail their checksum")

    context_rows = _rows(contexts_path)
    decisions: dict[str, dict[str, Any]] = {}
    # Complete every question-only decision before opening answers or metrics.
    for row in context_rows:
        if tuple(row["conditions"]) != CONDITION_IDS:
            raise ValueError(f"source conditions are invalid for {row['example_id']}")
        choice = selector.select(row["question"])
        decisions[row["example_id"]] = {
            "condition": choice.condition,
            "a": choice.a,
            "b": choice.b,
            "question_type": choice.question_type,
            "used_fallback": choice.used_fallback,
            "llm_call": dict(selector.last_call or {}),
        }
        print(f"{row['example_id']}: C8 -> {choice.condition}", flush=True)

    context_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-c8-contexts-", dir=context_dir.parent) as temp:
        temporary = Path(temp)
        output_contexts = temporary / "contexts.jsonl"
        with output_contexts.open("w", encoding="utf-8") as stream:
            for row in context_rows:
                decision = decisions[row["example_id"]]
                selected = decision["condition"]
                row["conditions"]["C8"] = {
                    **row["conditions"][selected],
                    "reused_from": selected,
                    "selection_seconds": decision["llm_call"].get("elapsed_seconds"),
                }
                row["llm_adaptive_selection"] = decision
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        context_manifest = {
            **source_context_manifest,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "conditions": list(NINE_CONDITION_IDS),
            "source_contexts_sha256": _hash(contexts_path),
            "contexts_sha256": _hash(output_contexts),
            "c8_selector": {
                "method": "zero-shot question-only LLM selection among C2-C6",
                "model": selector.client.model,
                "base_url": selector.client.base_url,
                "temperature": 0,
                "system_prompt": SYSTEM_PROMPT,
                "selection_counts": dict(sorted(Counter(
                    item["condition"] for item in decisions.values()
                ).items())),
                "selection_before_answer_access": True,
            },
            "adaptive_context_reuse": (
                source_context_manifest["adaptive_context_reuse"]
                + "; C8 copies the fixed arm chosen by the question-only LLM"
            ),
            "exploratory": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(context_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(context_dir)

    answer_rows = _rows(answers_path)
    if {row["example_id"] for row in answer_rows} != set(decisions):
        raise ValueError("source answers and contexts have different question IDs")
    totals: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="agp-c8-answers-", dir=answer_dir.parent) as temp:
        temporary = Path(temp)
        output_answers = temporary / "answers.jsonl"
        with output_answers.open("w", encoding="utf-8") as stream:
            for row in answer_rows:
                selected = decisions[row["example_id"]]["condition"]
                row["answers"]["C8"] = row["answers"][selected]
                row["automatic_metrics"]["C8"] = row["automatic_metrics"][selected]
                row["llm_calls"]["C8"] = {
                    "copied_from": selected, "new_model_call": False,
                }
                row["context_characters"]["C8"] = row["context_characters"][selected]
                row["llm_adaptive_reused_from"] = selected
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                for metric, value in row["automatic_metrics"]["C8"].items():
                    if isinstance(value, (int, float)) and value is not None:
                        totals.setdefault(metric, {"sum": 0.0})["sum"] += value
        count = len(answer_rows)
        c8_macro = {name: item["sum"] / count for name, item in totals.items()}
        answer_summary = {
            **source_answer_summary,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "conditions": list(NINE_CONDITION_IDS),
            "generation_calls_per_question": 7,
            "c8_answer_reuse": True,
            "c8_new_answer_calls": 0,
            "source_answers_sha256": _hash(answers_path),
            "answers_sha256": _hash(output_answers),
            "macro_average": {
                **source_answer_summary["macro_average"], "C8": c8_macro,
            },
            "c8_selection_counts": dict(sorted(Counter(
                item["condition"] for item in decisions.values()
            ).items())),
            "exploratory": True,
        }
        (temporary / "summary.json").write_text(
            json.dumps(answer_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(answer_dir)
    return context_manifest, answer_summary
