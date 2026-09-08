from pathlib import Path
import unittest

from agp_research.graph import KnowledgeGraph
from agp_research.models import Node


ROOT = Path(__file__).parents[1]


class GraphTest(unittest.TestCase):
    def test_load_and_exact_match(self):
        graph = KnowledgeGraph.from_csv(ROOT / "data/nodes.csv", ROOT / "data/edges.csv")
        matched, unmatched = graph.exact_match(["donald   trump", "Unknown Person"])
        self.assertEqual(matched, ["n1"])
        self.assertEqual(unmatched, ["Unknown Person"])

    def test_indexes_are_reused_after_graph_construction(self):
        class CountingNodes(dict):
            calls = 0

            def values(self):
                self.calls += 1
                return super().values()

        nodes = CountingNodes({"n1": Node("n1", "Census Australia", "")})
        graph = KnowledgeGraph(nodes=nodes, edges=[])
        self.assertEqual(nodes.calls, 1)

        self.assertEqual(graph.title_mentions("Census Australia?"),
                         [("Census Australia", 0, 16)])
        self.assertEqual(graph.exact_match(["census australia"]), (["n1"], []))
        self.assertEqual(nodes.calls, 1)

    def test_duplicate_title_behavior_remains_deterministic(self):
        graph = KnowledgeGraph(
            nodes={
                "first": Node("first", "Example", ""),
                "last": Node("last", "EXAMPLE", ""),
            },
            edges=[],
        )
        self.assertEqual(graph.title_mentions("Example"), [("Example", 0, 7)])
        self.assertEqual(graph.exact_match(["Example"]), (["last"], []))
