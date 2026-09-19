import unittest

from agp_research.eight_condition_retrieval import (
    AGP_STUDY_PARAMETERS,
    FIXED_AGP_PAIRS,
    NativeAGPBackendPool,
    direct_neighbours,
)
from agp_research.agp_native_backend import default_native_library
from agp_research.graph import KnowledgeGraph
from agp_research.models import Edge, Node


class FakeBackend:
    created = []

    def __init__(self, library, config):
        self.library = library
        self.config = config
        self.calls = []
        self.closed = False
        self.created.append(self)

    def propagate(self, graph, seeds, parameters):
        self.calls.append((graph, seeds, parameters))
        return [self.config.a, self.config.b]

    def close(self):
        self.closed = True


class EightConditionRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(
            nodes={key: Node(key, key.upper(), "") for key in "abcde"},
            edges=[
                Edge("a", "c"), Edge("b", "c"), Edge("b", "c"),
                Edge("a", "d"), Edge("b", "e"), Edge("a", "b"),
            ],
        )
        FakeBackend.created = []

    def test_direct_neighbours_deduplicate_edges_and_exclude_seeds(self):
        ranked = direct_neighbours(self.graph, ["a", "b", "missing"])
        self.assertEqual([item.node.id for item in ranked], ["c", "d", "e"])
        self.assertEqual([item.score for item in ranked], [2.0, 1.0, 1.0])
        self.assertEqual([item.node.id for item in direct_neighbours(self.graph, ["a", "b"], top_k=2)], ["c", "d"])

    def test_direct_neighbours_empty_and_invalid_budget(self):
        self.assertEqual(direct_neighbours(self.graph, []), [])
        with self.assertRaisesRegex(ValueError, "top_k"):
            direct_neighbours(self.graph, ["a"], top_k=0)

    def test_five_fixed_pairs_obey_paper_constraints(self):
        self.assertEqual(tuple(FIXED_AGP_PAIRS), ("C2", "C3", "C4", "C5", "C6"))
        self.assertEqual(len(set(FIXED_AGP_PAIRS.values())), 5)
        for a, b in FIXED_AGP_PAIRS.values():
            self.assertGreaterEqual(a + b, 1.0)

    def test_pool_reuses_backend_per_pair_and_fixes_other_parameters(self):
        with NativeAGPBackendPool(
            self.graph, "unused-library", backend_factory=FakeBackend
        ) as pool:
            self.assertEqual(pool.initialized_pairs, ())
            self.assertEqual(pool.query(["a"], (0.0, 1.0)), [0.0, 1.0])
            self.assertEqual(pool.query(["b"], (0.0, 1.0)), [0.0, 1.0])
            self.assertEqual(pool.query(["a"], (0.5, 0.5)), [0.5, 0.5])
            self.assertEqual(len(FakeBackend.created), 2)
            self.assertEqual(pool.initialized_pairs, ((0.0, 1.0), (0.5, 0.5)))
            self.assertTrue(all(item.config.query_type == "S" for item in FakeBackend.created))
            self.assertTrue(all(item.calls[0][2] == AGP_STUDY_PARAMETERS for item in FakeBackend.created))
            with self.assertRaisesRegex(ValueError, "one of"):
                pool.query(["a"], (0.3, 0.7))
        self.assertTrue(all(item.closed for item in FakeBackend.created))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.query(["a"], (0.0, 1.0))

    @unittest.skipUnless(default_native_library().is_file(), "native AGP library not built")
    def test_real_native_pairs_change_scores_on_irregular_graph(self):
        graph = KnowledgeGraph(
            nodes={str(index): Node(str(index), f"Node {index}", "") for index in range(6)},
            edges=[
                Edge("0", "1"), Edge("0", "2"), Edge("0", "3"),
                Edge("0", "4"), Edge("4", "5"),
            ],
        )
        with NativeAGPBackendPool(graph) as pool:
            scores = {}
            for condition, pair in FIXED_AGP_PAIRS.items():
                scores[condition] = {
                    item.node.id: item.score for item in pool.query(["5"], pair)
                }
            first_handle = pool._backends[(0.0, 1.0)]._handle
            pool.query(["0"], (0.0, 1.0))
            self.assertEqual(first_handle, pool._backends[(0.0, 1.0)]._handle)
            self.assertEqual(len(pool.initialized_pairs), 5)
        self.assertGreater(scores["C2"]["4"], scores["C6"]["4"])


if __name__ == "__main__":
    unittest.main()
