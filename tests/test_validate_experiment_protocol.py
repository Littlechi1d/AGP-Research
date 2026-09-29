import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_experiment_protocol import validate_protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(root: Path) -> Path:
    paths = {}
    for name in ("nodes", "edges", "questions", "rule", "native", "planner", "llm", "answers"):
        path = root / name
        path.write_text(name, encoding="utf-8")
        paths[name] = {"path": name, "sha256": _sha(path)}
    protocol = {
        "protocol_version": 1,
        "status": "frozen_before_cross_dataset_evaluation",
        "conditions": {
            "C0": "none", "C1": "neighbours",
            **{arm: {"a": pair[0], "b": pair[1]} for arm, pair in EXPECTED.items()},
            "C7": "rule", "C8": "llm",
        },
        "retrieval": {
            "depth": 2, "decay": 0.3, "top_k": 10, "query_type": "S",
            "relative_error": 0.1, "max_context_chars": None,
            "match_threshold": 0.85, "match_margin": 0.1,
        },
        "generation": {"temperature": 0, "max_answer_tokens": 256},
        "datasets": {"sample": {
            "nodes": paths["nodes"], "edges": paths["edges"],
            "questions": {**paths["questions"], "count": 40},
        }},
        "frozen_files": {
            "rule": paths["rule"], "native": paths["native"],
            "planner": paths["planner"], "llm": paths["llm"],
            "answers": paths["answers"],
        },
    }
    target = root / "protocol.json"
    target.write_text(json.dumps(protocol), encoding="utf-8")
    return target


EXPECTED = {
    "C2": (0.0, 1.0), "C3": (0.25, 0.75), "C4": (0.5, 0.5),
    "C5": (0.75, 0.25), "C6": (1.0, 0.0),
}


def test_accepts_frozen_protocol(tmp_path):
    result = validate_protocol(_protocol(tmp_path))
    assert result["valid"] is True
    assert result["checked_file_count"] == 8


def test_rejects_changed_frozen_input(tmp_path):
    protocol = _protocol(tmp_path)
    (tmp_path / "questions").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch.*questions"):
        validate_protocol(protocol)


def test_rejects_changed_setting(tmp_path):
    protocol = _protocol(tmp_path)
    content = json.loads(protocol.read_text(encoding="utf-8"))
    content["retrieval"]["top_k"] = 20
    protocol.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="retrieval settings"):
        validate_protocol(protocol)
