"""Shared data models for the AGP pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    id: str
    title: str
    description: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    description: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class AGPParameters:
    depth: int = 2
    decay: float = 0.6
    top_k: int = 10

    def validate(self) -> "AGPParameters":
        if not 0 <= self.depth <= 5:
            raise ValueError("depth must be between 0 and 5")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        return self


@dataclass(frozen=True)
class RankedNode:
    node: Node
    score: float


@dataclass
class PipelineResult:
    question: str
    strategy: str
    keywords: list[str]
    matched_seed_ids: list[str]
    unmatched_keywords: list[str]
    parameters: AGPParameters
    ranked_nodes: list[RankedNode]
    context: str
    answer: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

