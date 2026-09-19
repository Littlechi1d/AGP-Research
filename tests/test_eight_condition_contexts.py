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

    def test_character_cap_preserves_complete_ranked_lines(self):
        graph = KnowledgeGraph(
            nodes={
                "a": Node("a", "Alpha", "seed"),
                "b": Node("b", "Beta", "first " + "x" * 200),
                "c": Node("c", "Gamma", "second " + "y" * 200),
                "d": Node("d", "Delta", "third " + "z" * 200),
            },
            edges=[Edge("a", "b", "connects"), Edge("a", "c", "connects")],
        )
        ranked = [RankedNode(graph.nodes[key], 1.0) for key in ("b", "c", "d")]
        context = format_study_context(graph, ranked, ["a"], max_context_chars=512)
        self.assertLessEqual(len(context), 512)
        self.assertIn("Beta: first " + "x" * 200, context)
        self.assertIn("Gamma: second " + "y" * 200, context)
        self.assertNotIn("Delta:", context)
        self.assertNotIn("Delta", context)
        self.assertTrue(all(line.startswith("- ") or line.startswith("Retrieved")
                            for line in context.splitlines()))

    def test_character_cap_rejects_unfittable_first_entity(self):
        graph = KnowledgeGraph(
            nodes={"a": Node("a", "Alpha", "x" * 600)}, edges=[]
        )
        with self.assertRaisesRegex(ValueError, "highest-ranked"):
            format_study_context(graph, [RankedNode(graph.nodes["a"], 1.0)], [], 512)

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
                max_context_chars=512,
            )
            self.assertEqual(manifest["conditions"], list(CONDITION_IDS))
            self.assertEqual(manifest["max_context_chars"], 512)
            row = json.loads((output / "contexts.jsonl").read_text())
            self.assertEqual(row["matched_seed_ids"], ["a"])
            self.assertEqual(tuple(row["conditions"]), CONDITION_IDS)
            self.assertEqual(row["conditions"]["C0"]["context"], "")
            self.assertEqual(row["conditions"]["C0"]["context_characters"], 0)
            self.assertEqual(row["conditions"]["C1"]["ranked_node_ids"], ["b"])
            self.assertEqual(row["conditions"]["C1"]["context_characters"], len(row["conditions"]["C1"]["context"]))
            self.assertEqual(row["conditions"]["C1"]["context_node_ids"], ["b"])
            self.assertFalse(row["conditions"]["C1"]["context_truncated"])
            self.assertEqual(row["conditions"]["C7"]["reused_from"], "C3")
            self.assertEqual(row["conditions"]["C7"]["context"], row["conditions"]["C3"]["context"])
            self.assertTrue(all(arm["context_characters"] <= 512
                                for arm in row["conditions"].values()))
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

    def test_runner_marks_omitted_relationship_as_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = KnowledgeGraph(
                nodes={"a": Node("a", "Alpha", "seed"),
                       "b": Node("b", "Beta", "candidate")},
                edges=[Edge("a", "b", "relationship " + "x" * 500)],
            )
            questions = root / "questions.json"
            keywords = root / "keywords.jsonl"
            questions.write_text(json.dumps([{"id": "q1", "question": "Alpha neighbours"}]))
            keywords.write_text(json.dumps({"example_id": "q1", "keywords": ["Alpha"]}) + "\n")
            run_eight_condition_contexts(
                graph, questions, root / "study", self.selector, FakePool(graph),
                keyword_results_path=keywords, max_context_chars=512,
            )
            row = json.loads((root / "study" / "contexts.jsonl").read_text())
            condition = row["conditions"]["C1"]
            self.assertEqual(condition["context_node_ids"], ["b"])
            self.assertTrue(condition["context_truncated"])
            self.assertNotIn("relationship", condition["context"])


if __name__ == "__main__":
    unittest.main()
