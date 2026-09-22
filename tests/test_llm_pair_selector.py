import pytest

from agp_research.llm_pair_selector import LLMPairSelector


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.last_call = {"cache_hit": False}

    def complete_json(self, system, user):
        assert "C2" in system and "C6" in system
        assert user == "Which pages are related?"
        return self.response


def test_selects_only_declared_pair():
    selector = LLMPairSelector(FakeClient({"arm": "C4"}))
    choice = selector.select("Which pages are related?")
    assert (choice.condition, choice.a, choice.b) == ("C4", 0.5, 0.5)
    assert selector.last_call == {"cache_hit": False}


@pytest.mark.parametrize("response", [{"arm": "C9"}, {"arm": "C2", "score": 1}, {}])
def test_rejects_invalid_response(response):
    with pytest.raises(ValueError):
        LLMPairSelector(FakeClient(response)).select("Which pages are related?")
