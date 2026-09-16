import json
import tempfile
import unittest
from pathlib import Path

from agp_research.answer_evaluation import (
    answer_metrics,
    load_retrieval_contexts,
    run_answer_quality_study,
)
from agp_research.graph import KnowledgeGraph
from agp_research.models import Node


class FakeClient:
    model = "test-model"
    base_url = "local"

    def __init__(self):
        self.calls = []
        self.last_call = {}

    def complete_text(self, system, user):
        self.calls.append((system, user))
        self.last_call = {"cache_hit": False, "total_tokens": 10}
        return "Alpha" if len(self.calls) == 1 else "Alpha and Beta"


class AnswerEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(
            nodes={
                "a": Node("a", "Alpha", ""),
                "b": Node("b", "Beta", ""),
                "c": Node("c", "Gamma", ""),
            },
            edges=[],
        )

    def test_answer_metrics_score_entities_and_faithfulness(self):
        metrics = answer_metrics(
            "Alpha and Gamma", ["a", "b"], self.graph, context="Alpha and Beta"
        )
        self.assertEqual(metrics.entity_precision, 0.5)
        self.assertEqual(metrics.entity_recall, 0.5)
        self.assertEqual(metrics.entity_f1, 0.5)
        self.assertEqual(metrics.unsupported_node_ids, ["c"])
        self.assertEqual(metrics.context_faithfulness, 0.5)

    def test_answer_metrics_exclude_question_seeds_and_nested_titles(self):
        graph = KnowledgeGraph(
            nodes={
                "seed": Node("seed", "Alpha Base", ""),
                "nested": Node("nested", "Alpha", ""),
                "answer": Node("answer", "Beta", ""),
            },
            edges=[],
        )
        metrics = answer_metrics(
            "Alpha Base connects to Beta", ["answer"], graph,
            excluded_node_ids=["seed"],
        )
        self.assertEqual(metrics.mentioned_node_ids, ["answer"])
        self.assertEqual(metrics.entity_precision, 1.0)

    def test_context_loader_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            row = {"example_id": "q1", "strategy": "fixed", "context": "Alpha"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_retrieval_contexts(path, "fixed")

    def test_study_writes_paired_and_blinded_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            retrieval = root / "retrieval.jsonl"
            output = root / "study"
            questions.write_text(json.dumps([{
                "id": "q1", "question": "Name the pages", "question_type": "direct",
                "relevant_node_ids": ["a", "b"],
            }]))
            retrieval.write_text(json.dumps({
                "example_id": "q1", "strategy": "fixed", "context": "Alpha and Beta",
            }) + "\n")

            summary = run_answer_quality_study(
                FakeClient(), self.graph, questions, retrieval, output,
                retrieval_strategy="fixed",
            )

            self.assertEqual(summary["questions"], 1)
            self.assertTrue((output / "answers.jsonl").is_file())
            self.assertTrue((output / "blind_review.jsonl").is_file())
            self.assertTrue((output / "answer_key.json").is_file())
            self.assertTrue((output / "human_ratings.csv").is_file())
            record = json.loads((output / "answers.jsonl").read_text())
            self.assertEqual(record["metrics"]["agp_grounded"]["entity_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
