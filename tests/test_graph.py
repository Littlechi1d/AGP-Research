from pathlib import Path
import unittest

from agp_research.graph import KnowledgeGraph, edit_similarity
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

    def test_edit_similarity_is_normalized(self):
        self.assertEqual(edit_similarity("Census Australia", "census   australia"), 1.0)
        self.assertGreater(edit_similarity("Censuz Australia", "Census Australia"), 0.9)
        self.assertEqual(edit_similarity("", "Census Australia"), 0.0)

    def test_approximate_match_accepts_typo_and_preserves_evidence(self):
        graph = KnowledgeGraph(
            nodes={"n1": Node("n1", "Census Australia", "")}, edges=[]
        )
        matched, unmatched, details = graph.approximate_match(
            ["Censuz Australia"], threshold=0.8, margin=0.05
        )
        self.assertEqual(matched, ["n1"])
        self.assertEqual(unmatched, [])
        self.assertTrue(details[0].accepted)
        self.assertEqual(details[0].method, "approximate")
        self.assertEqual(details[0].matched_title, "Census Australia")

    def test_approximate_match_rejects_ambiguous_candidate(self):
        graph = KnowledgeGraph(
            nodes={
                "n1": Node("n1", "Example Alpha", ""),
                "n2": Node("n2", "Example Alphi", ""),
            },
            edges=[],
        )
        matched, unmatched, details = graph.approximate_match(
            ["Example Alphx"], threshold=0.8, margin=0.05
        )
        self.assertEqual(matched, [])
        self.assertEqual(unmatched, ["Example Alphx"])
        self.assertFalse(details[0].accepted)
        self.assertEqual(details[0].method, "unresolved")

    def test_approximate_match_recovers_split_compound_title(self):
        graph = KnowledgeGraph(
            nodes={"1": Node("1", "Chrisley Knows Best on USA", "")},
            edges=[],
        )

        matched, unmatched, details = graph.approximate_match(
            ["Chrisley Knows Best", "USA"], threshold=0.8, margin=0.05
        )

        self.assertEqual(matched, ["1"])
        self.assertEqual(unmatched, [])
        self.assertEqual(details[0].method, "approximate-compound")

    def test_approximate_match_validates_controls(self):
        graph = KnowledgeGraph(nodes={"n1": Node("n1", "Example", "")}, edges=[])
        with self.assertRaises(ValueError):
            graph.approximate_match(["Example"], threshold=1.1)
        with self.assertRaises(ValueError):
            graph.approximate_match(["Example"], margin=-0.1)
