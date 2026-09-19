"""Retrieval primitives for the proposed eight-condition AGP study."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from agp_research.agp_native_backend import NativeAGPBackend, default_native_library
from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, RankedNode
from agp_research.paper_backend import PaperBackendConfig


# The fixed arms and adaptive selector must use this same predeclared set.
FIXED_AGP_PAIRS: dict[str, tuple[float, float]] = {
    "C2": (0.00, 1.00),
    "C3": (0.25, 0.75),
    "C4": (0.50, 0.50),
    "C5": (0.75, 0.25),
    "C6": (1.00, 0.00),
}
AGP_STUDY_PARAMETERS = AGPParameters(depth=2, decay=0.30, top_k=10)


def direct_neighbours(
    graph: KnowledgeGraph,
    seed_ids: list[str],
    *,
    top_k: int = 10,
) -> list[RankedNode]:
    """Rank unique one-hop neighbours by shared-seed count, then stable node ID.

    Seeds are excluded. Duplicate edges do not increase a neighbour's score.
    The output contains no more than ``top_k`` nodes and uses no gold labels.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if any(edge.weight != 1.0 for edge in graph.edges):
        raise ValueError("eight-condition retrieval requires an unweighted graph")
    seeds = set(seed_ids) & graph.nodes.keys()
    if not seeds:
        return []

    adjacent_seeds: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        source, target = edge.source, edge.target
        if source == target:
            continue
        if source in seeds and target not in seeds:
            adjacent_seeds[target].add(source)
        if target in seeds and source not in seeds:
            adjacent_seeds[source].add(target)

    ranked = sorted(
        adjacent_seeds,
        key=lambda node_id: (-len(adjacent_seeds[node_id]), node_id),
    )
    return [
        RankedNode(graph.nodes[node_id], float(len(adjacent_seeds[node_id])))
        for node_id in ranked[:top_k]
    ]


class NativeAGPBackendPool:
    """Lazily keep one reusable native graph handle for each fixed `(a, b)` pair.

    Every query uses the same graph, PPR weights, result budget, error tolerance,
    and AGP-Static++ mode. The selected pair is the sole variable in C2–C7.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        library: str | Path = default_native_library(),
        *,
        backend_factory: Callable[[str | Path, PaperBackendConfig], NativeAGPBackend]
        = NativeAGPBackend,
    ) -> None:
        self.graph = graph
        self.library = Path(library)
        self._backend_factory = backend_factory
        self._backends: dict[tuple[float, float], NativeAGPBackend] = {}
        self._closed = False

    @property
    def initialized_pairs(self) -> tuple[tuple[float, float], ...]:
        """Pairs whose backend objects have been created; handles load on query."""
        return tuple(self._backends)

    def query(self, seed_ids: list[str], pair: tuple[float, float]) -> list[RankedNode]:
        """Query a fixed pair or a question-selected pair from the same set."""
        if self._closed:
            raise RuntimeError("native AGP backend pool is closed")
        if pair not in FIXED_AGP_PAIRS.values():
            raise ValueError(f"(a, b) must be one of {tuple(FIXED_AGP_PAIRS.values())}")
        if pair not in self._backends:
            config = PaperBackendConfig(
                a=pair[0], b=pair[1], query_type="S", relative_error=0.10
            ).validate()
            self._backends[pair] = self._backend_factory(self.library, config)
        return self._backends[pair].propagate(
            self.graph, seed_ids, AGP_STUDY_PARAMETERS
        )

    def close(self) -> None:
        if self._closed:
            return
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._closed = True

    def __enter__(self) -> "NativeAGPBackendPool":
        if self._closed:
            raise RuntimeError("native AGP backend pool is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
