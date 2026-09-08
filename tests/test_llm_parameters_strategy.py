"""Tests for the local-keywords plus LLM-parameters ablation."""

import unittest
from unittest.mock import patch

from agp_research.cli import parser
from agp_research.graph import KnowledgeGraph
from agp_research.models import Node
from agp_research.pipeline import AGPPipeline
from agp_research.planner import llm_parameters


class FakeClient:
    def __init__(self, response=None):
        self.response = response or {"depth": 2, "decay": 0.7, "top_k": 10}
        self.calls = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.response


class LLMParametersStrategyTest(unittest.TestCase):
    def setUp(self):
        title = (
            "Headquarters Marine Corps, Henderson Hall, "
            "Headquarters & Service Battalion"
        )
        self.graph = KnowledgeGraph(nodes={"seed": Node("seed", title, "")}, edges=[])
        self.question = f"Which pages directly neighbor {title}"

    def test_parameter_prompt_does_not_request_keywords(self):
        client = FakeClient()
        parameters = llm_parameters(self.question, client)
        self.assertEqual(parameters.__dict__, {"depth": 2, "decay": 0.7, "top_k": 10})
        self.assertNotIn("keywords", client.calls[0][0].casefold())
        self.assertEqual(client.calls[0][1], self.question)

    def test_invalid_top_k_is_rejected(self):
        client = FakeClient({"depth": 2, "decay": 0.7, "top_k": 7})
        with self.assertRaisesRegex(ValueError, "top_k must be 5, 10, or 20"):
            llm_parameters(self.question, client)

    def test_strategy_uses_complete_local_title_and_llm_parameters(self):
        client = FakeClient()
        result = AGPPipeline(self.graph, client).run(
            self.question, strategy="llm-parameters", generate_answer=False
        )
        self.assertEqual(result.keywords, [self.graph.nodes["seed"].title])
        self.assertEqual(result.matched_seed_ids, ["seed"])
        self.assertEqual(result.unmatched_keywords, [])
        self.assertEqual(result.parameters.__dict__, {"depth": 2, "decay": 0.7, "top_k": 10})
        self.assertTrue(result.metadata["llm_used"])
        self.assertEqual(len(client.calls), 1)

    def test_strategy_requires_client(self):
        with self.assertRaisesRegex(RuntimeError, "LLM-parameters strategy requires"):
            AGPPipeline(self.graph).run(
                self.question, strategy="llm-parameters", generate_answer=False
            )

    def test_cli_accepts_strategy_for_ask_and_experiment(self):
        with patch("sys.argv", ["agp", "ask", self.question,
                                "--strategy", "llm-parameters", "--no-answer"]):
            self.assertEqual(parser().parse_args().strategy, "llm-parameters")
        with patch("sys.argv", ["agp", "experiment",
                                "--strategies", "fixed", "llm-parameters"]):
            self.assertEqual(
                parser().parse_args().strategies, ["fixed", "llm-parameters"]
            )


if __name__ == "__main__":
    unittest.main()
