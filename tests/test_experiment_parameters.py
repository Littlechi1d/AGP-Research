"""Check parameter defaults, forwarding, validation, and CLI wiring."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agp_research.cli import main
from agp_research.evaluation import run_experiment
from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, Edge, Node
from agp_research.pipeline import AGPPipeline


class ExperimentParametersTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.questions = Path(self.temp.name) / "questions.json"
        self.output = Path(self.temp.name) / "results.jsonl"
        self.questions.write_text(json.dumps([
            {"id": "q1", "question": "Which pages neighbor Alpha", "relevant_node_ids": ["b"]}
        ]), encoding="utf-8")
        self.graph = KnowledgeGraph(
            nodes={key: Node(key, title, "") for key, title in
                   [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")]},
            edges=[Edge("a", "b"), Edge("b", "c")],
        )
        self.pipeline = AGPPipeline(self.graph, None)

    def records(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_default_configuration_is_preserved(self):
        run_experiment(self.pipeline, self.questions, self.output, ["fixed"])
        self.assertEqual(self.records()[0]["parameters"],
                         {"depth": 2, "decay": 0.6, "top_k": 10})

    def test_custom_parameters_reach_propagation_and_log(self):
        run_experiment(self.pipeline, self.questions, self.output, ["fixed"],
                       fixed_parameters=AGPParameters(1, 0.3, 2))
        record = self.records()[0]
        self.assertEqual(record["parameters"], {"depth": 1, "decay": 0.3, "top_k": 2})
        scores = {item["node"]["id"]: item["score"] for item in record["ranked_nodes"]}
        self.assertEqual(set(scores), {"a", "b"})
        self.assertAlmostEqual(scores["b"], 0.3)

    def test_invalid_configuration_preserves_existing_output(self):
        self.output.write_text("existing result", encoding="utf-8")
        for params in [AGPParameters(-1, 0.6, 10), AGPParameters(6, 0.6, 10),
                       AGPParameters(2, 1.1, 10), AGPParameters(2, 0.6, 0)]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                run_experiment(self.pipeline, self.questions, self.output,
                               ["fixed"], fixed_parameters=params)
        self.assertEqual(self.output.read_text(), "existing result")

    def test_rules_keep_their_own_parameters(self):
        run_experiment(self.pipeline, self.questions, self.output, ["fixed", "rules"],
                       fixed_parameters=AGPParameters(0, 0.2, 1))
        fixed, rules = self.records()
        self.assertEqual(fixed["parameters"]["depth"], 0)
        self.assertEqual(rules["parameters"], {"depth": 1, "decay": 0.4, "top_k": 10})

    def test_cli_passes_arguments_to_actual_runner(self):
        argv = ["agp", "experiment", "--questions", str(self.questions),
                "--output", str(self.output), "--strategies", "fixed",
                "--depth", "1", "--decay", "0.3", "--top-k", "2"]
        with patch("sys.argv", argv), \
             patch("agp_research.cli.KnowledgeGraph.from_csv", return_value=self.graph), \
             patch("agp_research.cli.OpenAICompatibleClient.from_environment", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(self.records()[0]["parameters"],
                         {"depth": 1, "decay": 0.3, "top_k": 2})
