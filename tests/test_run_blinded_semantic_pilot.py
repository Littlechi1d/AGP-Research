import pytest

from scripts.run_blinded_semantic_pilot import _validate_score, select_question_ids


def test_selects_one_question_per_type_deterministically():
    questions = [
        {"id": f"{kind}_{number}", "question_type": kind}
        for kind in ("direct", "comparison", "path", "similarity")
        for number in range(3)
    ]
    selected = select_question_ids(questions)
    assert len(selected) == 4
    assert selected == select_question_ids(list(reversed(questions)))
    assert {value.rsplit("_", 1)[0] for value in selected} == {
        "direct", "comparison", "path", "similarity"
    }


def test_validates_judge_scores():
    assert _validate_score({
        "correctness": 5, "completeness": 4, "reason": "Mostly complete."
    })["reason"] == "Mostly complete."
    with pytest.raises(ValueError, match="correctness"):
        _validate_score({"correctness": 0, "completeness": 4, "reason": "Bad"})
    with pytest.raises(ValueError, match="reason"):
        _validate_score({"correctness": 3, "completeness": 3, "reason": ""})
