import hashlib
import json

from scripts.evaluate_llm_selector_from_contexts import evaluate


class FakeClient:
    model = "fake"
    base_url = "local"

    def __init__(self):
        self.last_call = {}

    def complete_json(self, system, user):
        self.last_call = {"cache_hit": False}
        return {"arm": "C3"}


def test_evaluates_selected_arm_from_saved_development_contexts(tmp_path):
    questions = tmp_path / "questions_dev.json"
    questions.write_text(json.dumps([{
        "id": "q1", "question": "Question one", "question_type": "direct",
        "relevant_node_ids": ["n1"],
    }]))
    context_dir = tmp_path / "contexts"
    context_dir.mkdir()
    context_file = context_dir / "contexts.jsonl"
    conditions = {
        arm: {"ranked_node_ids": ["n1"] if arm == "C3" else ["other"]}
        for arm in ("C2", "C3", "C4", "C5", "C6")
    }
    context_file.write_text(json.dumps({
        "example_id": "q1", "question": "Question one", "conditions": conditions,
    }) + "\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (context_dir / "manifest.json").write_text(json.dumps({
        "contexts_sha256": digest(context_file),
        "input_sha256": {"questions": digest(questions)},
    }))

    summary = evaluate(
        FakeClient(), questions, context_file, tmp_path / "output"
    )
    assert summary["selection_counts"] == {"C3": 1}
    assert summary["llm_selected_mean_retrieval_f1"] == 1.0
    assert summary["mean_retrieval_f1_by_fixed_arm"]["C2"] == 0.0
