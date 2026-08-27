"""Keyword extraction and query-dependent AGP parameter selection."""

from __future__ import annotations

import re

from agp_research.graph import KnowledgeGraph, normalize
from agp_research.llm import OpenAICompatibleClient
from agp_research.models import AGPParameters


def local_keyword_extraction(question: str, graph: KnowledgeGraph) -> list[str]:
    """Find complete node titles, treating punctuation as a valid boundary."""
    normalized_question = normalize(question)
    matches = [
        node.title
        for node in graph.nodes.values()
        if re.search(
            rf"(?<!\w){re.escape(normalize(node.title))}(?!\w)",
            normalized_question,
        )
    ]
    return sorted(matches, key=lambda text: (-len(text), text))


def rule_parameters(question: str, *, top_k: int = 10) -> AGPParameters:
    text = normalize(question)
    multi_hop = ("connection", "connect", "through", "chain", "indirect", "path")
    explanatory = ("why", "how did", "explain", "effect", "impact", "influence")
    comparison = ("compare", "difference", "similar", "both", "versus", " vs ")
    if any(term in text for term in multi_hop):
        return AGPParameters(depth=3, decay=0.75, top_k=top_k)
    if any(term in text for term in explanatory):
        return AGPParameters(depth=2, decay=0.7, top_k=top_k)
    if any(term in f" {text} " for term in comparison):
        return AGPParameters(depth=2, decay=0.6, top_k=top_k)
    return AGPParameters(depth=1, decay=0.4, top_k=top_k)


def llm_plan(question: str, client: OpenAICompatibleClient) -> tuple[list[str], AGPParameters]:
    system = """You configure graph retrieval. Extract entity keywords exactly as written
in the question and choose propagation parameters. Use depth 1 for direct facts, 2
for explanation/comparison, and 3 for multi-step connections. decay controls distant
evidence and must be between 0 and 1. top_k must be 5, 10, or 20. Return JSON only
with keys keywords, depth, decay, and top_k."""
    data = client.complete_json(system, question)
    keywords = [str(item).strip() for item in data.get("keywords", []) if str(item).strip()]
    parameters = AGPParameters(
        depth=int(data["depth"]),
        decay=float(data["decay"]),
        top_k=int(data["top_k"]),
    ).validate()
    if parameters.top_k not in {5, 10, 20}:
        raise ValueError("LLM top_k must be 5, 10, or 20")
    return keywords, parameters


def heuristic_keywords(question: str) -> list[str]:
    """Last-resort noun-like tokens, useful for reporting unmatched terms."""
    stop = {"what", "when", "where", "which", "who", "why", "how", "is", "did", "does", "the", "and", "are", "was", "were", "with", "from", "into"}
    return [token for token in re.findall(r"[A-Za-z][A-Za-z'-]+", question) if token.casefold() not in stop]
