import json
import tempfile
import unittest
from pathlib import Path

from agp_research.frozen_evaluation import (
    FROZEN_CONDITIONS,
    FrozenCondition,
    _paired_bootstrap,
    run_frozen_evaluation,
)
from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, Edge, Node
from agp_research.pipeline import AGPPipeline


class FrozenEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = self.root / "questions.json"
        self.questions.write_text(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "question": "Which pages directly neighbor Alpha",
                        "question_type": "direct",
                        "relevant_node_ids": ["b"],
                    },
                    {
                        "id": "q2",
                        "question": "Which pages directly neighbor Beta",
                        "question_type": "direct",
                        "relevant_node_ids": ["a", "c"],
                    },
                ]
            ),
            encoding="utf-8",
        )
        graph = KnowledgeGraph(
            nodes={
                node_id: Node(node_id, title, "")
                for node_id, title in (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"))
            },
            edges=[Edge("a", "b"), Edge("b", "c")],
        )
        self.pipeline = AGPPipeline(graph)
        self.conditions = (
            FrozenCondition("fixed-k1", "fixed", AGPParameters(1, 0.3, 1)),
            FrozenCondition("rules", "rules"),
        )

    def test_writes_complete_named_bundle(self):
        output = self.root / "run"
        summary = run_frozen_evaluation(
            self.pipeline,
            self.questions,
            output,
            conditions=self.conditions,
            bootstrap_samples=100,
        )
        records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 4)
        self.assertEqual(set(summary), {"fixed-k1", "rules"})
        self.assertIn("f1", records[0]["metrics"])
        self.assertEqual(records[0]["condition"], "fixed-k1")
        for name in (
            "summary.json",
            "by_question_type.json",
            "paired_comparisons.json",
            "manifest.json",
        ):
            self.assertTrue((output / name).is_file())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["record_count"], 4)
        self.assertEqual(manifest["bootstrap_seed"], 90055)

    def test_frozen_conditions_match_predeclared_protocol(self):
        self.assertEqual(
            [condition.name for condition in FROZEN_CONDITIONS],
            [
                "seed-only",
                "fixed-k5",
                "fixed-k10",
                "rules",
                "llm-keywords",
                "llm-parameters",
                "llm-parameters-few-shot",
                "llm",
            ],
        )
        by_name = {condition.name: condition for condition in FROZEN_CONDITIONS}
        self.assertEqual(by_name["fixed-k5"].parameters, AGPParameters(2, 0.3, 5))
        self.assertEqual(by_name["fixed-k10"].parameters, AGPParameters(2, 0.3, 10))
        self.assertEqual(
            by_name["llm-keywords"].parameters, AGPParameters(2, 0.3, 5)
        )

    def test_refuses_to_overwrite_output(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            run_frozen_evaluation(
                self.pipeline, self.questions, output, conditions=self.conditions
            )
        self.assertEqual(marker.read_text(), "keep")

    def test_paired_bootstrap_is_deterministic(self):
        records = [
            {"condition": condition, "example_id": question, "metrics": {"f1": score}}
            for condition, values in (("rules", (0.5, 0.5)), ("candidate", (0.7, 0.3)))
            for question, score in zip(("q1", "q2"), values)
        ]
        first = _paired_bootstrap(records, samples=100, seed=7)
        second = _paired_bootstrap(records, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["candidate"]["mean_f1_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
