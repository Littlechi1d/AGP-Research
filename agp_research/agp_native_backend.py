"""Persistent ctypes interface to the AGP-Dynamic C++ query implementation."""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, RankedNode
from agp_research.paper_backend import PaperBackendConfig


def default_native_library() -> Path:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return Path(f"build/paper_backend/libagp_api{suffix}")


class NativeAGPBackend:
    """Load one graph into C++ once, then issue repeated AGP queries in-process."""

    def __init__(
        self,
        library: str | Path = default_native_library(),
        config: PaperBackendConfig | None = None,
    ):
        self._handle: int | None = None
        self._graph_identity: int | None = None
        self._node_ids: list[str] = []
        self._one_based: dict[str, int] = {}
        self.library_path = Path(library).expanduser().resolve()
        self.config = (config or PaperBackendConfig()).validate()
        if not self.library_path.is_file():
            raise FileNotFoundError(f"AGP native library not found: {self.library_path}")
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_api()

    def _configure_api(self) -> None:
        self._library.agp_create.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
        ]
        self._library.agp_create.restype = ctypes.c_void_p
        self._library.agp_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_char,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint,
        ]
        self._library.agp_query.restype = ctypes.c_int
        self._library.agp_vertex_count.argtypes = [ctypes.c_void_p]
        self._library.agp_vertex_count.restype = ctypes.c_uint
        self._library.agp_last_error.argtypes = []
        self._library.agp_last_error.restype = ctypes.c_char_p
        self._library.agp_destroy.argtypes = [ctypes.c_void_p]
        self._library.agp_destroy.restype = None

    def _error(self) -> str:
        message = self._library.agp_last_error()
        return message.decode(errors="replace") if message else "unknown native error"

    def _load_graph(self, graph: KnowledgeGraph) -> None:
        if not graph.nodes:
            raise ValueError("native AGP requires a nonempty graph")
        if any(edge.weight != 1.0 for edge in graph.edges):
            raise ValueError("the AGP reference implementation supports unweighted graphs only")

        self._node_ids = list(graph.nodes)
        self._one_based = {
            node_id: index for index, node_id in enumerate(self._node_ids, start=1)
        }
        simple_edges: set[tuple[int, int]] = set()
        degrees = [0] * len(self._node_ids)
        for edge in graph.edges:
            if edge.source == edge.target:
                continue
            source, target = sorted(
                (self._one_based[edge.source], self._one_based[edge.target])
            )
            simple_edges.add((source, target))
        endpoints = [endpoint for edge in sorted(simple_edges) for endpoint in edge]
        for source, target in simple_edges:
            degrees[source - 1] += 1
            degrees[target - 1] += 1
        if self.config.query_type == "S" and any(degree == 0 for degree in degrees):
            raise ValueError("AGP-Static++ cannot initialize isolated vertices")

        endpoint_array = (ctypes.c_int * len(endpoints))(*endpoints)
        handle = self._library.agp_create(
            len(self._node_ids),
            endpoint_array,
            len(endpoints),
            self.config.a,
            self.config.b,
            int(self.config.query_type == "S"),
        )
        if not handle:
            raise RuntimeError(f"could not create native AGP graph: {self._error()}")
        self._handle = handle
        self._graph_identity = id(graph)

    @property
    def loaded(self) -> bool:
        return self._handle is not None

    def close(self) -> None:
        if self._handle is not None:
            self._library.agp_destroy(self._handle)
            self._handle = None
            self._graph_identity = None

    def __enter__(self) -> "NativeAGPBackend":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def propagate(
        self,
        graph: KnowledgeGraph,
        seed_ids: list[str],
        parameters: AGPParameters,
    ) -> list[RankedNode]:
        parameters.validate()
        if self._handle is None:
            self._load_graph(graph)
        elif self._graph_identity != id(graph):
            raise ValueError("a NativeAGPBackend instance cannot be reused with another graph")

        seeds = list(
            dict.fromkeys(seed for seed in seed_ids if seed in self._one_based)
        )
        if not seeds:
            return []
        continuation = parameters.decay
        restart = 1.0 - continuation
        if restart <= 0.0:
            raise ValueError("paper PPR weights require decay < 1")
        weights = [
            restart * continuation**hop for hop in range(parameters.depth + 1)
        ]
        delta = self.config.delta or 1.0 / len(self._node_ids)
        epsilon = (
            self.config.relative_error**2
            * delta
            / (parameters.depth + 1)
            / 2.0
        )
        seed_array = (ctypes.c_int * len(seeds))(
            *(self._one_based[seed] for seed in seeds)
        )
        mass_array = (ctypes.c_double * len(seeds))(
            *(1.0 / len(seeds) for _ in seeds)
        )
        weight_array = (ctypes.c_double * len(weights))(*weights)
        output = (ctypes.c_double * len(self._node_ids))()
        status = self._library.agp_query(
            self._handle,
            seed_array,
            mass_array,
            len(seeds),
            parameters.depth,
            weight_array,
            epsilon,
            self.config.query_type.encode("ascii"),
            output,
            len(self._node_ids),
        )
        if status != 0:
            raise RuntimeError(f"native AGP query failed ({status}): {self._error()}")
        ranked = sorted(
            (
                (index, score)
                for index, score in enumerate(output)
                if score > 0.0 and math.isfinite(score)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RankedNode(graph.nodes[self._node_ids[index]], score)
            for index, score in ranked[: parameters.top_k]
        ]
