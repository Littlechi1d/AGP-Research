import unittest
import json
import tempfile
from pathlib import Path

from agp_research.agp_pair_selector import (
    QuestionTypePairSelector,
    best_fixed_arm,
    leave_one_out_validation,
    question_type_from_text,
    train_question_type_selector,
)
from agp_research.eight_condition_retrieval import FIXED_AGP_PAIRS


class AGPPairSelectorTest(unittest.TestCase):
    def test_recognizes_four_templates_and_falls_back(self):
        self.assertEqual(question_type_from_text("Which pages directly neighbor Alpha"), "direct")
        self.assertEqual(question_type_from_text("Which pages are liked by both A and B"), "comparison")
        self.assertEqual(question_type_from_text("Which pages form the unique shortest path between A and B"), "path")
        self.assertEqual(question_type_from_text("Which pages similar to A are two hops away"), "similarity")
        self.assertIsNone(question_type_from_text("Tell me about Alpha"))

    def test_selector_uses_only_question_wording(self):
        selector = QuestionTypePairSelector(
            {"direct": "C2", "comparison": "C3", "path": "C4", "similarity": "C5"},
            "C6",
        )
        self.assertEqual(selector.select("Which pages directly neighbor Alpha").condition, "C2")
        fallback = selector.select("Tell me about Alpha")
        self.assertEqual(fallback.condition, "C6")
        self.assertTrue(fallback.used_fallback)
        self.assertEqual((fallback.a, fallback.b), FIXED_AGP_PAIRS["C6"])

    def test_selector_loads_saved_development_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            path.write_text(json.dumps({
                "arm_by_type": {
                    "direct": "C2", "comparison": "C3", "path": "C4", "similarity": "C5"
                },
                "fallback_arm": "C6",
            }))
            selector = QuestionTypePairSelector.from_json(path)
            self.assertEqual(selector.select("Which pages directly neighbor A").condition, "C2")

    def test_training_selects_highest_mean_and_stable_tie(self):
        rows = [
            {"example_id": "d1", "question": "Which pages directly neighbor A", "question_type": "direct",
             "f1_by_arm": {"C2": .2, "C3": .8, "C4": .1, "C5": 0, "C6": 0}},
            {"example_id": "d2", "question": "Which pages directly neighbor B", "question_type": "direct",
             "f1_by_arm": {"C2": .2, "C3": .6, "C4": .1, "C5": 0, "C6": 0}},
            {"example_id": "p1", "question": "Which pages form the shortest path between A and B", "question_type": "path",
             "f1_by_arm": {"C2": .5, "C3": .5, "C4": .1, "C5": 0, "C6": 0}},
        ]
        self.assertEqual(best_fixed_arm(rows), "C3")
        selector = train_question_type_selector(rows)
        self.assertEqual(selector.arm_by_type["direct"], "C3")
        self.assertEqual(selector.arm_by_type["path"], "C2")
        self.assertEqual(selector.arm_by_type["comparison"], "C3")

    def test_leave_one_out_never_uses_heldout_scores_for_training(self):
        rows = [
            {"example_id": "d1", "question": "Which pages directly neighbor A", "question_type": "direct",
             "f1_by_arm": {"C2": 1, "C3": 0, "C4": 0, "C5": 0, "C6": 0}},
            {"example_id": "d2", "question": "Which pages directly neighbor B", "question_type": "direct",
             "f1_by_arm": {"C2": 0, "C3": 1, "C4": 0, "C5": 0, "C6": 0}},
        ]
        validation = leave_one_out_validation(rows)
        self.assertEqual(validation["question_count"], 2)
        self.assertEqual(validation["adaptive_mean_f1"], 0.0)
        self.assertEqual([row["selected_arm"] for row in validation["decisions"]], ["C3", "C2"])


if __name__ == "__main__":
    unittest.main()
