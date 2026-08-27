import unittest

from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, Edge, Node
from agp_research.propagation import propagate


class PropagationTest(unittest.TestCase):
    def test_propagation_reaches_two_hop_node(self):
        graph = KnowledgeGraph(
            nodes={key: Node(key, key.upper(), key) for key in ("a", "b", "c")},
            edges=[Edge("a", "b"), Edge("b", "c")],
        )
        ranked = propagate(graph, ["a"], AGPParameters(depth=2, decay=0.5, top_k=3))
        scores = {item.node.id: item.score for item in ranked}
        self.assertEqual(scores["a"], 1.125)
        self.assertEqual(scores["b"], 0.5)
        self.assertEqual(scores["c"], 0.125)

    def test_no_seed_returns_no_results(self):
        graph = KnowledgeGraph(nodes={"a": Node("a", "A", "")}, edges=[])
        self.assertEqual(propagate(graph, [], AGPParameters()), [])
