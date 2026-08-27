"""Adapter for the reference implementation from the AGP-Dynamic paper.

The upstream repository is benchmark code rather than a library. This module
exports our graph to its binary format, invokes a small C++ bridge, and maps the
returned contiguous integer vertex IDs back to application node IDs.
"""

from __future__ import annotations

import math
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, RankedNode


@dataclass(frozen=True)
class PaperBackendConfig:
    """Parameters required by the paper's approximate query algorithm."""

    a: float = 0.0
    b: float = 1.0
    delta: float | None = None
    relative_error: float = 0.1
    query_type: str = "S"

    def validate(self) -> "PaperBackendConfig":
        if not 0.0 <= self.a <= 1.0 or not 0.0 <= self.b <= 1.0:
            raise ValueError("a and b must be in [0, 1]")
        if self.a + self.b < 1.0:
            raise ValueError("the paper requires a + b >= 1")
        if self.delta is not None and self.delta <= 0.0:
            raise ValueError("delta must be positive")
        if self.relative_error <= 0.0:
            raise ValueError("relative_error must be positive")
        if self.query_type not in {"N", "S"}:
            raise ValueError("query_type must be N (exact) or S (AGP-Static++)")
        return self


class PaperAGPBackend:
    """Call the authors' C++ propagation implementation through a subprocess."""

    def __init__(self, executable: str | Path, config: PaperBackendConfig | None = None):
        self.executable = Path(executable).expanduser().resolve()
        self.config = (config or PaperBackendConfig()).validate()
        if not self.executable.is_file():
            raise FileNotFoundError(f"AGP bridge executable not found: {self.executable}")

    def propagate(
        self,
        graph: KnowledgeGraph,
        seed_ids: list[str],
        parameters: AGPParameters,
    ) -> list[RankedNode]:
        parameters.validate()
        if not graph.nodes or not seed_ids:
            return []
        if any(edge.weight != 1.0 for edge in graph.edges):
            raise ValueError("the paper implementation supports unweighted graphs only")

        node_ids = list(graph.nodes)
        one_based = {node_id: index for index, node_id in enumerate(node_ids, 1)}
        seeds = list(dict.fromkeys(node_id for node_id in seed_ids if node_id in one_based))
        if not seeds:
            return []

        degrees = {node_id: 0 for node_id in node_ids}
        simple_edges: set[tuple[int, int]] = set()
        for edge in graph.edges:
            if edge.source == edge.target:
                continue
            u, v = sorted((one_based[edge.source], one_based[edge.target]))
            simple_edges.add((u, v))
        for u, v in simple_edges:
            degrees[node_ids[u - 1]] += 1
            degrees[node_ids[v - 1]] += 1
        isolated = [node_id for node_id, degree in degrees.items() if degree == 0]
        if self.config.query_type == "S" and isolated:
            raise ValueError(
                "AGP-Static++ cannot initialize isolated vertices in the reference code; "
                f"remove them or use query_type='N': {isolated[:5]}"
            )

        continuation = parameters.decay
        restart = 1.0 - continuation
        if restart <= 0.0:
            raise ValueError("paper PPR weights require decay < 1")
        weights = [restart * continuation**hop for hop in range(parameters.depth + 1)]
        delta = self.config.delta or 1.0 / len(node_ids)
        epsilon = (
            self.config.relative_error**2
            * delta
            / (parameters.depth + 1)
            / 2.0
        )

        with tempfile.TemporaryDirectory(prefix="agp-paper-") as directory:
            root = Path(directory)
            graph_path = root / "graph.bin"
            query_path = root / "query.txt"
            self._write_graph(graph_path, len(node_ids), sorted(simple_edges))
            self._write_query(
                query_path,
                seeds=[one_based[node_id] for node_id in seeds],
                a=self.config.a,
                b=self.config.b,
                epsilon=epsilon,
                weights=weights,
                query_type=self.config.query_type,
            )
            completed = subprocess.run(
                [str(self.executable), str(graph_path), str(query_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        scores: dict[int, float] = {}
        for line in completed.stdout.splitlines():
            if not line.startswith("SCORE\t"):
                continue
            _, raw_id, raw_score = line.split("\t")
            score = float(raw_score)
            if math.isfinite(score):
                scores[int(raw_id)] = score
        if len(scores) != len(node_ids):
            raise RuntimeError(
                f"AGP bridge returned {len(scores)} scores for {len(node_ids)} nodes; "
                f"stderr: {completed.stderr.strip()}"
            )
        ranked = sorted(
            ((index, score) for index, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RankedNode(graph.nodes[node_ids[index - 1]], score)
            for index, score in ranked[: parameters.top_k]
        ]

    @staticmethod
    def _write_graph(path: Path, node_count: int, edges: list[tuple[int, int]]) -> None:
        endpoints = [endpoint for edge in edges for endpoint in edge]
        with path.open("wb") as stream:
            stream.write(struct.pack("=II", node_count, len(endpoints)))
            if endpoints:
                stream.write(struct.pack(f"={len(endpoints)}i", *endpoints))

    @staticmethod
    def _write_query(
        path: Path,
        *,
        seeds: list[int],
        a: float,
        b: float,
        epsilon: float,
        weights: list[float],
        query_type: str,
    ) -> None:
        seed_mass = 1.0 / len(seeds)
        lines = [
            f"{a:.17g} {b:.17g} {len(weights) - 1} {epsilon:.17g} {query_type}",
            " ".join(f"{weight:.17g}" for weight in weights),
            str(len(seeds)),
            *(f"{seed} {seed_mass:.17g}" for seed in seeds),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
