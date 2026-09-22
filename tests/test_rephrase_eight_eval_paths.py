import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.rephrase_eight_eval_paths import rephrase_path_candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RephraseEightEvalPathsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nodes = self.root / "nodes.csv"
        self.edges = self.root / "edges.csv"
        self.nodes.write_text(
            "id,title\na,Alpha\nb,Beta\nc,Gamma\nd,Delta\n",
            encoding="utf-8",
        )
        self.edges.write_text(
            "source,target\na,b\nb,c\nc,d\n", encoding="utf-8"
        )
        self.previous = self.root / "previous"
        self.previous.mkdir()
        questions = [{
            "id": "path_01", "question_type": "path",
            "question": "Which two intermediate pages connect Alpha to Delta?",
            "seed_node_ids": ["a", "d"], "relevant_node_ids": ["b", "c"],
        }, {
            "id": "direct_01", "question_type": "direct",
            "question": "Which pages directly neighbor Alpha?",
            "seed_node_ids": ["a"], "relevant_node_ids": ["b"],
        }]
        question_file = self.previous / "questions.json"
        question_file.write_text(json.dumps(questions), encoding="utf-8")
        review_file = self.previous / "criteria_review.csv"
        with review_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "example_id", "question", "reviewer_id", "notes",
                "question_unambiguous_yes_no", "labels_correct_yes_no",
            ])
            writer.writeheader()
            for item in questions:
                writer.writerow({"example_id": item["id"], "question": item["question"]})
        (self.previous / "manifest.json").write_text(json.dumps({
            "questions_sha256": digest(question_file),
            "criteria_review_sha256": digest(review_file),
            "source_sha256": {"nodes": digest(self.nodes), "edges": digest(self.edges)},
            "status": "candidate_pending_independent_criteria_review",
        }), encoding="utf-8")

    def test_only_path_wording_changes_and_hashes_match(self):
        output = self.root / "revised"
        manifest = rephrase_path_candidate(
            self.previous, self.nodes, self.edges, output
        )
        rows = json.loads((output / "questions.json").read_text())
        self.assertEqual(rows[0]["relevant_node_ids"], ["b", "c"])
        self.assertIn("using the fewest page-to-page connections", rows[0]["question"])
        self.assertIn("which two pages do you pass through", rows[0]["question"])
        self.assertEqual(rows[1]["question"], "Which pages directly neighbor Alpha?")
        self.assertEqual(manifest["rephrased_path_examples"], ["path_01"])
        self.assertEqual(manifest["questions_sha256"], digest(output / "questions.json"))
        self.assertEqual(manifest["criteria_review_sha256"], digest(output / "criteria_review.csv"))
        with (output / "criteria_review.csv").open(newline="") as stream:
            review = list(csv.DictReader(stream))
        self.assertEqual(review[0]["question"], rows[0]["question"])
        self.assertEqual(review[0]["reviewer_id"], "")

    def test_rejects_wrong_path_answer_order(self):
        file = self.previous / "questions.json"
        rows = json.loads(file.read_text())
        rows[0]["relevant_node_ids"] = ["c", "b"]
        file.write_text(json.dumps(rows), encoding="utf-8")
        manifest_file = self.previous / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["questions_sha256"] = digest(file)
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "answer order"):
            rephrase_path_candidate(
                self.previous, self.nodes, self.edges, self.root / "revised"
            )
        self.assertFalse((self.root / "revised").exists())


if __name__ == "__main__":
    unittest.main()
