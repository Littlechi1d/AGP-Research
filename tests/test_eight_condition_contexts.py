import json
import tempfile
import unittest
from pathlib import Path

from agp_research.agp_pair_selector import QuestionTypePairSelector
from agp_research.eight_condition_contexts import (
    CONDITION_IDS,
    format_study_context,
    load_saved_keywords,
    run_eight_condition_contexts,
)
from agp_research.graph import KnowledgeGraph
from agp_research.models import Edge, Node, RankedNode


class FakePool:
    def __init__(self, graph):
        self.graph = graph
        self.calls = []

    def query(self, seed_ids, pair):
        self.calls.append((tuple(seed_ids), pair))
        return [RankedNode(self.graph.nodes["b"], 0.5)]


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.last_call = {"model": "fake", "cache_hit": False}

    def complete_json(self, system, user):
        self.calls += 1
        return {"keywords": ["Alpha"]}


class EightConditionContextsTest(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(
            nodes={
                "a": Node("a", "Alpha", "seed"),
                "b": Node("b", "Beta", "candidate"),
                "c": Node("c", "Gamma", "other"),
            },
            edges=[Edge("a", "b", "connects")],
        )
        self.selector = QuestionTypePairSelector(
            {"direct": "C3", "comparison": "C2", "path": "C2", "similarity": "C2"},
            "C2",
        )

    def test_formatter_shows_seed_to_candidate_edge_without_extra_entity(self):
        context = format_study_context(
            self.graph, [RankedNode(self.graph.nodes["b"], 1.0)], ["a"]
        )
        self.assertIn("Beta: candidate", context)
        self.assertIn("Alpha -> Beta: connects", context)
        self.assertNotIn("Alpha: seed", context)
        self.assertNotIn("score=", context)

    def test_saved_keyword_loader_rejects_duplicate_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keywords.jsonl"
            row = {"example_id": "q1", "keywords": ["Alpha"]}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_saved_keywords(path)

    def test_runner_uses_same_seeds_and_reuses_selected_fixed_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            keywords = root / "keywords.jsonl"
            output = root / "study"
            questions.write_text(json.dumps([{
                "id": "q1", "question": "Which pages directly neighbor Alpha",
                "question_type": "direct", "relevant_node_ids": ["c"],
            }]))
            keywords.write_text(json.dumps({"example_id": "q1", "keywords": ["Alpha"]}) + "\n")
            pool = FakePool(self.graph)
            manifest = run_eight_condition_contexts(
                self.graph, questions, output, self.selector, pool,
                keyword_results_path=keywords,
            )
            self.assertEqual(manifest["conditions"], list(CONDITION_IDS))
            row = json.loads((output / "contexts.jsonl").read_text())
            self.assertEqual(row["matched_seed_ids"], ["a"])
            self.assertEqual(tuple(row["conditions"]), CONDITION_IDS)
            self.assertEqual(row["conditions"]["C0"]["context"], "")
            self.assertEqual(row["conditions"]["C0"]["context_characters"], 0)
            self.assertEqual(row["conditions"]["C1"]["ranked_node_ids"], ["b"])
            self.assertEqual(row["conditions"]["C1"]["context_characters"], len(row["conditions"]["C1"]["context"]))
            self.assertEqual(row["conditions"]["C7"]["reused_from"], "C3")
            self.assertEqual(row["conditions"]["C7"]["context"], row["conditions"]["C3"]["context"])
            self.assertEqual(len(pool.calls), 5)
            self.assertTrue(all(call[0] == ("a",) for call in pool.calls))
            self.assertNotIn("relevant_node_ids", row)
            with self.assertRaises(FileExistsError):
                run_eight_condition_contexts(
                    self.graph, questions, output, self.selector, pool,
                    keyword_results_path=keywords,
                )

    def test_runner_calls_live_keyword_model_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            questions.write_text(json.dumps([{
                "id": "q1", "question": "Which pages directly neighbor Alpha",
            }]))
            client = FakeClient()
            run_eight_condition_contexts(
                self.graph, questions, root / "study", self.selector,
                FakePool(self.graph), client=client,
            )
            self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
