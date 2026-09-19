"""Generate the eight predeclared retrieval contexts before answer generation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from agp_research.agp_pair_selector import QuestionTypePairSelector
from agp_research.eight_condition_retrieval import (
    AGP_STUDY_PARAMETERS,
    FIXED_AGP_PAIRS,
    NativeAGPBackendPool,
    direct_neighbours,
)
from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient
from agp_research.models import RankedNode
from agp_research.planner import llm_keywords


CONDITION_IDS = ("C0", "C1", *FIXED_AGP_PAIRS, "C7")


class AGPPool(Protocol):
    def query(self, seed_ids: list[str], pair: tuple[float, float]) -> list[RankedNode]: ...


def format_study_context(
    graph: KnowledgeGraph,
    ranked_nodes: list[RankedNode],
    seed_ids: list[str],
) -> str:
    """Use identical evidence formatting for all seven graph-backed arms.

    Seed-to-result edges are shown even when the seed is not itself in the ranked
    top 10. This lets the direct-neighbour arm expose *why* a page was retrieved.
    """
    if not ranked_nodes:
        return "No graph evidence was retrieved."
    selected = {item.node.id for item in ranked_nodes}
    seeds = set(seed_ids) & graph.nodes.keys()
    lines = ["Retrieved entities:"]
    for item in ranked_nodes:
        lines.append(f"- {item.node.title}: {item.node.description}")

    relationships = []
    seen: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        source, target = edge.source, edge.target
        if source == target or not (
            (source in selected and target in selected | seeds)
            or (target in selected and source in seeds)
        ):
            continue
        key = (min(source, target), max(source, target), edge.description)
        if key not in seen:
            seen.add(key)
            relationships.append(edge)
    if relationships:
        lines.append("Retrieved relationships:")
        for edge in relationships:
            lines.append(
                f"- {graph.nodes[edge.source].title} -> "
                f"{graph.nodes[edge.target].title}: {edge.description}"
            )
    return "\n".join(lines)


def load_saved_keywords(path: str | Path) -> dict[str, list[str]]:
    """Read one previously generated LLM keyword list per example ID."""
    saved: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row["example_id"]
        if example_id in saved:
            raise ValueError(f"duplicate saved keywords for {example_id}")
        saved[example_id] = row["keywords"]
    return saved


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(
    graph: KnowledgeGraph,
    ranked_nodes: list[RankedNode],
    seed_ids: list[str],
    elapsed_seconds: float,
    *,
    pair: tuple[float, float] | None = None,
    reused_from: str | None = None,
) -> dict[str, Any]:
    if len(ranked_nodes) > AGP_STUDY_PARAMETERS.top_k:
        raise ValueError("retriever exceeded the shared 10-node context budget")
    context = format_study_context(graph, ranked_nodes, seed_ids)
    return {
        "ranked_node_ids": [item.node.id for item in ranked_nodes],
        "scores": [item.score for item in ranked_nodes],
        "context": context,
        "context_characters": len(context),
        "query_seconds": elapsed_seconds,
        "pair": list(pair) if pair is not None else None,
        "reused_from": reused_from,
    }


def run_eight_condition_contexts(
    graph: KnowledgeGraph,
    questions_path: str | Path,
    output_dir: str | Path,
    selector: QuestionTypePairSelector,
    pool: AGPPool,
    *,
    client: OpenAICompatibleClient | None = None,
    keyword_results_path: str | Path | None = None,
    input_paths: dict[str, str | Path] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Save C0–C7 contexts atomically; never use relevance labels for retrieval."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if keyword_results_path is None and client is None:
        raise ValueError("provide saved LLM keywords or an LLM client")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    questions_file = Path(questions_path)
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique")
    saved = load_saved_keywords(keyword_results_path) if keyword_results_path else None
    if saved is not None:
        missing = sorted(set(ids) - set(saved))
        if missing:
            raise ValueError(f"saved keywords missing questions: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agp-eight-contexts-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        result_path = temporary_path / "contexts.jsonl"
        with result_path.open("w", encoding="utf-8") as stream:
            for example in questions:
                question = example["question"]
                started = time.perf_counter()
                if saved is None:
                    assert client is not None
                    keywords = llm_keywords(question, client)
                    extraction_call = dict(client.last_call)
                    keyword_source = "live_llm"
                else:
                    keywords = saved[example["id"]]
                    extraction_call = None
                    keyword_source = "saved_llm"
                extraction_seconds = time.perf_counter() - started
                seed_ids, unmatched, matches = graph.approximate_match(
                    keywords, threshold=0.85, margin=0.10
                )
                # Select from question wording before running or inspecting AGP arms.
                started = time.perf_counter()
                selection = selector.select(question)
                selector_seconds = time.perf_counter() - started
                conditions: dict[str, dict[str, Any]] = {
                    "C0": {
                        "ranked_node_ids": [], "scores": [], "context": "",
                        "context_characters": 0,
                        "query_seconds": 0.0, "pair": None, "reused_from": None,
                    }
                }

                started = time.perf_counter()
                neighbours = direct_neighbours(
                    graph, seed_ids, top_k=AGP_STUDY_PARAMETERS.top_k
                )
                conditions["C1"] = _condition(
                    graph, neighbours, seed_ids, time.perf_counter() - started
                )
                for arm, pair in FIXED_AGP_PAIRS.items():
                    started = time.perf_counter()
                    ranked = pool.query(seed_ids, pair)
                    conditions[arm] = _condition(
                        graph, ranked, seed_ids, time.perf_counter() - started,
                        pair=pair,
                    )

                chosen = conditions[selection.condition]
                conditions["C7"] = {
                    **chosen,
                    "reused_from": selection.condition,
                    "selection_seconds": selector_seconds,
                }
                if tuple(conditions) != CONDITION_IDS:
                    raise RuntimeError("eight-condition output is incomplete")
                stream.write(json.dumps({
                    "example_id": example["id"],
                    "question": question,
                    "question_type": example.get("question_type"),
                    "keywords": keywords,
                    "keyword_source": keyword_source,
                    "keyword_extraction_seconds": extraction_seconds,
                    "keyword_extraction_call": extraction_call,
                    "matched_seed_ids": seed_ids,
                    "unmatched_keywords": unmatched,
                    "keyword_matches": [asdict(item) for item in matches],
                    "adaptive_selection": asdict(selection),
                    "conditions": conditions,
                }, ensure_ascii=False) + "\n")

        hashes = {"questions": _hash(questions_file)}
        if keyword_results_path is not None:
            hashes["saved_keywords"] = _hash(Path(keyword_results_path))
        for name, path in (input_paths or {}).items():
            hashes[name] = _hash(Path(path))
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "conditions": list(CONDITION_IDS),
            "fixed_pairs": FIXED_AGP_PAIRS,
            "agp_parameters": asdict(AGP_STUDY_PARAMETERS),
            "query_type": "S",
            "relative_error": 0.10,
            "match_threshold": 0.85,
            "match_margin": 0.10,
            "keyword_source": "saved_llm" if saved is not None else "live_llm",
            "input_sha256": hashes,
            "contexts_sha256": _hash(result_path),
            "retrieval_only": True,
            "adaptive_context_reuse": (
                "C7 selects before AGP queries and copies the chosen fixed arm's "
                "ranking/context; its query_seconds is that arm's measured query "
                "time, not an additional executed query"
            ),
            "timing_caveat": "first query for each pair includes lazy native graph initialization",
        }
        (temporary_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(output)
    return manifest
