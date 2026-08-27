"""Matrix-free adaptive graph propagation."""

from __future__ import annotations

from collections import defaultdict

from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, RankedNode


def propagate(
    graph: KnowledgeGraph,
    seed_ids: list[str],
    parameters: AGPParameters,
) -> list[RankedNode]:
    """Calculate pi = sum(decay**i * M**i * x) on an undirected graph.

    M is a row-normalized, weighted adjacency operator. Seed mass is distributed
    equally. A separate frontier is propagated at every hop, while accumulated
    relevance is retained in the final score.
    """

    # Validate parameters and filter seed IDs to those present in the graph
    parameters.validate()
    valid_seeds = list(dict.fromkeys(seed for seed in seed_ids if seed in graph.nodes))
    if not valid_seeds:
        return []

    # Build a row-normalized adjacency dictionary for the graph
    adjacency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for edge in graph.edges:
        weight = max(edge.weight, 0.0)  # Ignore negative weights
        if edge.source == edge.target or weight == 0.0:  # Ignore self-loops and zero-weight edges
            continue
        adjacency[edge.source][edge.target] += weight
        adjacency[edge.target][edge.source] += weight

    # Initialize the propagation process with equal mass for each seed node
    seed_mass = 1.0 / len(valid_seeds)
    frontier = {seed: seed_mass for seed in valid_seeds}
    scores = defaultdict(float, frontier)

    # Iteratively propagate the mass through the graph for the specified depth
    for hop in range(1, parameters.depth + 1):
        next_frontier: dict[str, float] = defaultdict(float)
        for source, mass in frontier.items():
            total_weight = sum(adjacency[source].values())
            if total_weight == 0.0:
                continue
            for target, weight in adjacency[source].items():
                next_frontier[target] += mass * weight / total_weight
        hop_weight = parameters.decay**hop
        for node_id, mass in next_frontier.items():
            scores[node_id] += hop_weight * mass
        frontier = dict(next_frontier)

    # Rank the nodes based on their accumulated scores and return the top_k results
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [RankedNode(graph.nodes[node_id], score) for node_id, score in ranked[: parameters.top_k]]

