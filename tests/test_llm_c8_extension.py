import json
from pathlib import Path

from agp_research.llm_c8_extension import add_llm_c8
from agp_research.llm_pair_selector import LLMPairSelector


class FakeClient:
    model = "fake"
    base_url = "local"
    last_call = {"elapsed_seconds": 0.1, "cache_hit": False}

    def complete_json(self, system, user):
        return {"arm": "C4"}


def test_c8_reuses_llm_selected_fixed_context_and_answer(tmp_path: Path):
    contexts_dir = tmp_path / "source_contexts"
    answers_dir = tmp_path / "source_answers"
    contexts_dir.mkdir()
    answers_dir.mkdir()
    fixed = {"context": "evidence", "ranked_node_ids": ["n"], "context_characters": 8}
    conditions = {
        "C0": {"context": "", "ranked_node_ids": [], "context_characters": 0},
        "C1": fixed,
        **{f"C{i}": fixed for i in range(2, 7)},
        "C7": {**fixed, "reused_from": "C2"},
    }
    context_file = contexts_dir / "contexts.jsonl"
    context_file.write_text(json.dumps({
        "example_id": "q1", "question": "A question", "conditions": conditions,
    }) + "\n")
    import hashlib
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (contexts_dir / "manifest.json").write_text(json.dumps({
        "contexts_sha256": digest(context_file), "adaptive_context_reuse": "C7 reuse",
    }))
    answer_file = answers_dir / "answers.jsonl"
    answer_file.write_text(json.dumps({
        "example_id": "q1",
        "answers": {f"C{i}": f"answer {i}" for i in range(8)},
        "automatic_metrics": {f"C{i}": {"entity_f1": i / 10} for i in range(8)},
        "llm_calls": {f"C{i}": {} for i in range(8)},
        "context_characters": {f"C{i}": i for i in range(8)},
    }) + "\n")
    (answers_dir / "summary.json").write_text(json.dumps({
        "answers_sha256": digest(answer_file), "question_count": 1,
        "macro_average": {f"C{i}": {"entity_f1": i / 10} for i in range(8)},
    }))

    add_llm_c8(
        LLMPairSelector(FakeClient()), context_file, answer_file,
        tmp_path / "contexts9", tmp_path / "answers9",
    )
    context = json.loads((tmp_path / "contexts9" / "contexts.jsonl").read_text())
    answer = json.loads((tmp_path / "answers9" / "answers.jsonl").read_text())
    assert context["conditions"]["C8"]["reused_from"] == "C4"
    assert answer["answers"]["C8"] == "answer 4"
    assert answer["automatic_metrics"]["C8"] == {"entity_f1": 0.4}
