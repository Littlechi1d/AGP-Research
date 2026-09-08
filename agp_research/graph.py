"""Knowledge-graph loading and exact entity matching."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from agp_research.models import Edge, Node


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


@dataclass
class KnowledgeGraph:
    nodes: dict[str, Node]
    edges: list[Edge]
    _node_id_by_title: dict[str, str] = field(init=False, repr=False)
    _display_title_by_title: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build reusable lookup indexes once per loaded graph."""
        self._node_id_by_title = {}
        self._display_title_by_title = {}
        for node in self.nodes.values():
            normalized_title = normalize(node.title)
            # Preserve exact_match's previous last-duplicate-wins behavior.
            self._node_id_by_title[normalized_title] = node.id
            # Preserve extraction's previous first-duplicate-wins behavior.
            self._display_title_by_title.setdefault(normalized_title, node.title)

    @classmethod
    def from_csv(cls, nodes_path: str | Path, edges_path: str | Path) -> "KnowledgeGraph":
        with Path(nodes_path).open(encoding="utf-8", newline="") as stream:
            nodes = {
                row["id"]: Node(row["id"], row["title"], row.get("description", ""))
                for row in csv.DictReader(stream)
            }
        with Path(edges_path).open(encoding="utf-8", newline="") as stream:
            edges = [
                Edge(
                    source=row["source"],
                    target=row["target"],
                    description=row.get("description", ""),
                    weight=float(row.get("weight", 1.0)),
                )
                for row in csv.DictReader(stream)
            ]
        unknown = {
            endpoint
            for edge in edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in nodes
        }
        if unknown:
            raise ValueError(f"Edges refer to unknown node IDs: {sorted(unknown)}")
        return cls(nodes=nodes, edges=edges)

    def exact_match(self, keywords: list[str]) -> tuple[list[str], list[str]]:
        matched: list[str] = []
        unmatched: list[str] = []
        for keyword in keywords:
            node_id = self._node_id_by_title.get(normalize(keyword))
            if node_id is None:
                unmatched.append(keyword)
            elif node_id not in matched:
                matched.append(node_id)
        return matched, unmatched

    def title_mentions(self, text: str) -> list[tuple[str, int, int]]:
        """Find indexed titles at the same Unicode word boundaries as ``\\w``."""
        normalized_text = normalize(text)
        length = len(normalized_text)
        is_word = [
            character == "_" or character.isalnum()
            for character in normalized_text
        ]
        starts = [
            position
            for position in range(length)
            if (position == 0 or not is_word[position - 1])
            and not normalized_text[position].isspace()
        ]
        ends = [
            position
            for position in range(1, length + 1)
            if (position == length or not is_word[position])
            and not normalized_text[position - 1].isspace()
        ]

        mentions: list[tuple[str, int, int]] = []
        for start in starts:
            for end in ends:
                if end <= start:
                    continue
                normalized_title = normalized_text[start:end]
                title = self._display_title_by_title.get(normalized_title)
                if title is not None:
                    mentions.append((title, start, end))
        return mentions
