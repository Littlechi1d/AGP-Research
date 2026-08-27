"""Knowledge-graph loading and exact entity matching."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from agp_research.models import Edge, Node


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


@dataclass
class KnowledgeGraph:
    nodes: dict[str, Node]
    edges: list[Edge]

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
        title_index = {normalize(node.title): node.id for node in self.nodes.values()}
        matched: list[str] = []
        unmatched: list[str] = []
        for keyword in keywords:
            node_id = title_index.get(normalize(keyword))
            if node_id is None:
                unmatched.append(keyword)
            elif node_id not in matched:
                matched.append(node_id)
        return matched, unmatched

