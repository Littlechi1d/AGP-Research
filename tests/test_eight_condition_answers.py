import csv
import json
import tempfile
import unittest
from pathlib import Path

from agp_research.agp_pair_selector import QuestionTypePairSelector
from agp_research.eight_condition_answers import (
    load_eight_contexts,
    run_eight_condition_answers,
)
from agp_research.eight_condition_contexts import run_eight_condition_contexts
from agp_research.graph import KnowledgeGraph
from agp_research.models import Edge, Node, RankedNode


class FakePool:
    def __init__(self, graph):
        self.graph = graph

    def query(self, seed_ids, pair):
        return [RankedNode(self.graph.nodes["b"], 0.5)]


class FakeClient:
    model = "fake-model"
    base_url = "local"

    def __init__(self):
        self.calls = []
        self.last_call = {}

    def complete_text(self, system, user):
        self.calls.append((system, user))
        self.last_call = {
            "network_request": True, "cache_hit": False,
            "prompt_tokens": 10, "completion_tokens": 2,
        }
        return "Beta"


class EightConditionAnswersTest(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(
            nodes={
                "a": Node("a", "Alpha", "seed"),
                "b": Node("b", "Beta", "answer"),
            },
            edges=[Edge("a", "b", "connects")],
        )
        self.selector = QuestionTypePairSelector(
            {"direct": "C3", "comparison": "C2", "path": "C2", "similarity": "C2"},
            "C2",
        )

    def _inputs(self, root):
        questions = root / "questions.json"
        keywords = root / "keywords.jsonl"
        contexts = root / "contexts"
        questions.write_text(json.dumps([{
            "id": "q1", "question": "Which pages directly neighbor Alpha",
            "question_type": "direct", "relevant_node_ids": ["b"],
        }]))
        keywords.write_text(json.dumps({"example_id": "q1", "keywords": ["Alpha"]}) + "\n")
        run_eight_condition_contexts(
            self.graph, questions, contexts, self.selector, FakePool(self.graph),
            keyword_results_path=keywords,
        )
        return questions, contexts

    def test_eight_answers_blind_files_and_c7_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions, contexts = self._inputs(root)
            output = root / "answers"
            client = FakeClient()
            summary = run_eight_condition_answers(
                client, self.graph, questions, contexts / "contexts.jsonl", output
            )
            self.assertEqual(len(client.calls), 7)
            self.assertEqual(summary["question_count"], 1)
            self.assertEqual(summary["network_requests"], 7)
            row = json.loads((output / "answers.jsonl").read_text())
            self.assertEqual(tuple(row["answers"]), tuple(f"C{i}" for i in range(8)))
            self.assertEqual(row["adaptive_reused_from"], "C3")
            self.assertEqual(row["answers"]["C7"], row["answers"]["C3"])
            self.assertEqual(row["llm_calls"]["C7"]["new_model_call"], False)
            self.assertEqual(row["automatic_metrics"]["C1"]["entity_recall"], 1.0)

            blind = json.loads((output / "blind_review.jsonl").read_text())
            key = json.loads((output / "answer_key.json").read_text())
            self.assertEqual(set(blind["answers"]), set("ABCDEFGH"))
            self.assertEqual(set(key["q1"].values()), {f"C{i}" for i in range(8)})
            self.assertNotIn("C7", json.dumps(blind))
            with (output / "human_ratings.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.reader(stream))), 9)
            with (output / "faithfulness_ratings.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.reader(stream))), 8)
            with self.assertRaises(FileExistsError):
                run_eight_condition_answers(
                    client, self.graph, questions, contexts / "contexts.jsonl", output
                )

    def test_changed_context_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions, contexts = self._inputs(root)
            context_file = contexts / "contexts.jsonl"
            context_file.write_text(context_file.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_eight_condition_answers(
                    FakeClient(), self.graph, questions, context_file, root / "answers"
                )
            self.assertFalse((root / "answers").exists())

    def test_changed_question_labels_are_rejected_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions, contexts = self._inputs(root)
            data = json.loads(questions.read_text())
            data[0]["relevant_node_ids"] = ["a"]
            questions.write_text(json.dumps(data))
            client = FakeClient()
            with self.assertRaisesRegex(ValueError, "questions do not match"):
                run_eight_condition_answers(
                    client, self.graph, questions, contexts / "contexts.jsonl",
                    root / "answers",
                )
            self.assertEqual(client.calls, [])

    def test_context_loader_rejects_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _questions, contexts = self._inputs(root)
            context_file = contexts / "contexts.jsonl"
            row = context_file.read_text()
            context_file.write_text(row + row)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_eight_contexts(context_file)


if __name__ == "__main__":
    unittest.main()
