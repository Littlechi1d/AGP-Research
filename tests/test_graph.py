from pathlib import Path
import unittest

from agp_research.graph import KnowledgeGraph


ROOT = Path(__file__).parents[1]


class GraphTest(unittest.TestCase):
    def test_load_and_exact_match(self):
        graph = KnowledgeGraph.from_csv(ROOT / "data/nodes.csv", ROOT / "data/edges.csv")
        matched, unmatched = graph.exact_match(["donald   trump", "Unknown Person"])
        self.assertEqual(matched, ["n1"])
        self.assertEqual(unmatched, ["Unknown Person"])
