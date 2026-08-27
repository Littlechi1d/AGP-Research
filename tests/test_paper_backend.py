import tempfile
import unittest
from pathlib import Path

from agp_research.paper_backend import PaperAGPBackend, PaperBackendConfig


class PaperBackendTest(unittest.TestCase):
    def test_parameter_constraints(self):
        with self.assertRaises(ValueError):
            PaperBackendConfig(a=0.2, b=0.2).validate()

    def test_graph_binary_header_and_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.bin"
            PaperAGPBackend._write_graph(path, 3, [(1, 2), (2, 3)])
            self.assertEqual(len(path.read_bytes()), 8 + 4 * 4)

    def test_query_file_contains_normalized_seed_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.txt"
            PaperAGPBackend._write_query(
                path,
                seeds=[1, 3],
                a=0.0,
                b=1.0,
                epsilon=0.001,
                weights=[0.2, 0.16],
                query_type="S",
            )
            lines = path.read_text().splitlines()
            self.assertEqual(lines[-2:], ["1 0.5", "3 0.5"])

