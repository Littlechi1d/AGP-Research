import sys
import unittest
from pathlib import Path

from agp_research.agp_native_backend import NativeAGPBackend, default_native_library
from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, Edge, Node
from agp_research.paper_backend import PaperAGPBackend, PaperBackendConfig


class NativeAGPBackendTest(unittest.TestCase):
    def test_default_library_has_platform_suffix(self):
        expected = ".dylib" if sys.platform == "darwin" else ".so"
        self.assertEqual(default_native_library().suffix, expected)

    def test_missing_library_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "native library"):
            NativeAGPBackend("does-not-exist/libagp_api.so")


@unittest.skipUnless(default_native_library().is_file(), "native AGP library not built")
class NativeAGPIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(
            nodes={str(index): Node(str(index), f"Node {index}", "") for index in range(4)},
            edges=[Edge("0", "1"), Edge("1", "2"), Edge("2", "3")],
        )
        self.parameters = AGPParameters(depth=2, decay=0.6, top_k=4)

    def test_repeated_exact_queries_reuse_one_native_graph(self):
        backend = NativeAGPBackend(
            default_native_library(), PaperBackendConfig(query_type="N")
        )
        try:
            first = backend.propagate(self.graph, ["0"], self.parameters)
            handle = backend._handle
            second = backend.propagate(self.graph, ["3"], self.parameters)
            self.assertTrue(backend.loaded)
            self.assertEqual(backend._handle, handle)
            self.assertEqual(len(first), 3)
            self.assertEqual(len(second), 3)
        finally:
            backend.close()
        self.assertFalse(backend.loaded)

    def test_native_exact_scores_match_one_shot_bridge(self):
        executable = Path("build/paper_backend/agp_query_bridge")
        if not executable.is_file():
            self.skipTest("one-shot bridge not built")
        config = PaperBackendConfig(query_type="N")
        native = NativeAGPBackend(default_native_library(), config)
        one_shot = PaperAGPBackend(executable, config)
        try:
            native_result = native.propagate(self.graph, ["0"], self.parameters)
            bridge_result = one_shot.propagate(self.graph, ["0"], self.parameters)
        finally:
            native.close()
        self.assertEqual(
            [item.node.id for item in native_result],
            [item.node.id for item in bridge_result],
        )
        for left, right in zip(native_result, bridge_result):
            self.assertAlmostEqual(left.score, right.score, places=12)

    def test_repeated_approximate_queries_reuse_prepared_graph(self):
        backend = NativeAGPBackend(default_native_library())
        try:
            first = backend.propagate(self.graph, ["0"], self.parameters)
            handle = backend._handle
            second = backend.propagate(self.graph, ["3"], self.parameters)
            self.assertEqual(backend._handle, handle)
            self.assertTrue(first)
            self.assertTrue(second)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
