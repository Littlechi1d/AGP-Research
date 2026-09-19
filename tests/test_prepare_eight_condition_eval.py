import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_eight_condition_eval import prepare_eval_candidate


class PrepareEightConditionEvalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nodes = self.root / "nodes.csv"
        self.edges = self.root / "edges.csv"
        self.targets = self.root / "targets.csv"
        for path in (self.nodes, self.edges, self.targets):
            path.write_text("fixture\n")
        self.prior = self.root / "prior.json"
        self.prior.write_text(json.dumps([{
            "id": "old", "question": "Old question",
            "seed_node_ids": ["a"],
        }]))
        self.titles = {"a": "Alpha", "b": "Beta", "c": "Gamma"}
        self.adjacency = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
        self.categories = {key: "category" for key in self.titles}

    def _prepare(self, candidate):
        with patch("scripts.prepare_eight_condition_eval.read_graph",
                   return_value=(self.titles, self.adjacency)), \
             patch("scripts.prepare_eight_condition_eval.read_categories",
                   return_value=self.categories), \
             patch("scripts.prepare_eight_condition_eval.generate_split",
                   return_value=[candidate]) as generate, \
             patch("scripts.prepare_eight_condition_eval.validate"):
            manifest = prepare_eval_candidate(
                self.nodes, self.edges, self.targets, [self.prior],
                self.root / "candidate", per_type=1,
            )
        self.assertEqual(generate.call_args.args[4], {"a"})
        return manifest

    def test_creates_disjoint_candidate_and_blank_review_form(self):
        candidate = {
            "id": "facebook_direct_01", "question": "New question about Beta",
            "question_type": "direct", "seed_node_ids": ["b"],
            "relevant_node_ids": ["c"], "generation_rule": "Fixture rule",
        }
        manifest = self._prepare(candidate)
        self.assertEqual(manifest["status"], "candidate_pending_independent_criteria_review")
        self.assertEqual(manifest["seed_overlap_with_prior"], 0)
        saved = json.loads((self.root / "candidate" / "questions.json").read_text())
        self.assertEqual(saved[0]["id"], "eight_eval_direct_01")
        with (self.root / "candidate" / "criteria_review.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["expected_answer_titles"], "Gamma")
        self.assertEqual(rows[0]["labels_correct_yes_no"], "")

    def test_rejects_seed_overlap_before_writing(self):
        candidate = {
            "id": "facebook_direct_01", "question": "New question about Alpha",
            "question_type": "direct", "seed_node_ids": ["a"],
            "relevant_node_ids": ["b"], "generation_rule": "Fixture rule",
        }
        with self.assertRaisesRegex(AssertionError, "overlap"):
            self._prepare(candidate)
        self.assertFalse((self.root / "candidate").exists())

    def test_similarity_question_uses_plain_language(self):
        candidate = {
            "id": "facebook_similarity_01",
            "question": "Which pages similar to Beta share its category and are two hops away",
            "question_type": "similarity", "seed_node_ids": ["b"],
            "relevant_node_ids": ["c"], "generation_rule": "Fixture rule",
        }
        self._prepare(candidate)
        saved = json.loads((self.root / "candidate" / "questions.json").read_text())
        question = saved[0]["question"]
        self.assertIn("same type as Beta", question)
        self.assertIn("through one other page", question)
        self.assertIn("aren't directly connected", question)
        self.assertNotIn("two hops", question)

    def test_requires_prior_questions(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            prepare_eval_candidate(
                self.nodes, self.edges, self.targets, [], self.root / "candidate"
            )


if __name__ == "__main__":
    unittest.main()
