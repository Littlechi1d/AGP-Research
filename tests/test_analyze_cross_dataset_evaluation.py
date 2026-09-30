import pytest

from scripts.analyze_cross_dataset_evaluation import paired_bootstrap, retrieval_f1


def test_retrieval_f1_uses_sets():
    assert retrieval_f1(["a", "b", "b"], ["b", "c"]) == pytest.approx(0.5)
    assert retrieval_f1([], ["a"]) == 0.0


def test_paired_bootstrap_is_deterministic():
    result = paired_bootstrap([1.0, 0.5, 0.0], [0.0, 0.5, 1.0], samples=1000)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result == paired_bootstrap(
        [1.0, 0.5, 0.0], [0.0, 0.5, 1.0], samples=1000
    )
    assert result["ci95_low"] <= 0 <= result["ci95_high"]


def test_paired_bootstrap_validates_inputs():
    with pytest.raises(ValueError, match="same nonzero length"):
        paired_bootstrap([], [])
    with pytest.raises(ValueError, match="same nonzero length"):
        paired_bootstrap([1.0], [1.0, 2.0])
