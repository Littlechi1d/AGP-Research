"""Experimental question-only LLM selection among the five frozen AGP pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agp_research.agp_pair_selector import PairSelection
from agp_research.eight_condition_retrieval import FIXED_AGP_PAIRS
from agp_research.llm import OpenAICompatibleClient


SYSTEM_PROMPT = """Choose one graph-propagation setting for retrieving context for a question.
You see only the question, not the graph, relevance labels, or retrieval scores.
Available settings are C2: a=0,b=1; C3: a=0.25,b=0.75;
C4: a=0.5,b=0.5; C5: a=0.75,b=0.25; C6: a=1,b=0.
The settings change how graph propagation weights nodes of different degree.
There is no guaranteed best choice from wording alone. Make your best prediction.
Respond with JSON containing exactly one key, "arm", whose value is C2, C3, C4, C5, or C6."""


@dataclass
class LLMPairSelector:
    client: OpenAICompatibleClient
    last_call: dict[str, Any] | None = None

    def select(self, question: str) -> PairSelection:
        if not question.strip():
            raise ValueError("question must not be empty")
        response = self.client.complete_json(SYSTEM_PROMPT, question)
        self.last_call = dict(self.client.last_call)
        if not isinstance(response, dict) or set(response) != {"arm"}:
            raise ValueError(f"LLM selector returned invalid response: {response!r}")
        arm = response["arm"]
        if not isinstance(arm, str) or arm not in FIXED_AGP_PAIRS:
            raise ValueError(f"LLM selector returned unknown arm: {arm!r}")
        a, b = FIXED_AGP_PAIRS[arm]
        return PairSelection(arm, a, b, None, False)
