import unittest
from unittest.mock import patch

from agp_research.cli import parser
from agp_research.graph import KnowledgeGraph
from agp_research.models import AGPParameters, Node
from agp_research.pipeline import AGPPipeline
from agp_research.planner import llm_keywords


class FakeClient:
    def __init__(self, response=None):
        self.response = response or {"keywords": ["Gluten-Free Girl"]}
        self.calls = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.response


class LLMKeywordsStrategyTest(unittest.TestCase):
    def setUp(self):
        self.question = "Which pages directly neighbor Gluten-Free Girl"
        self.graph = KnowledgeGraph(
            nodes={"seed": Node("seed", "Gluten-Free Girl", "")}, edges=[]
        )

    def test_keyword_prompt_does_not_request_parameters(self):
        client = FakeClient()
        self.assertEqual(llm_keywords(self.question, client), ["Gluten-Free Girl"])
        prompt = client.calls[0][0].casefold()
        self.assertNotIn("depth", prompt)
        self.assertNotIn("decay", prompt)
        self.assertNotIn("top_k", prompt)
        self.assertEqual(client.calls[0][1], self.question)

    def test_empty_keywords_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one keyword"):
            llm_keywords(self.question, FakeClient({"keywords": []}))

    def test_strategy_uses_llm_keywords_and_supplied_fixed_parameters(self):
        parameters = AGPParameters(depth=2, decay=0.3, top_k=5)
        client = FakeClient()
        result = AGPPipeline(self.graph, client).run(
            self.question,
            strategy="llm-keywords",
            fixed_parameters=parameters,
            generate_answer=False,
        )
        self.assertEqual(result.keywords, ["Gluten-Free Girl"])
        self.assertEqual(result.matched_seed_ids, ["seed"])
        self.assertEqual(result.parameters, parameters)
        self.assertTrue(result.metadata["llm_used"])

    def test_strategy_requires_client(self):
        with self.assertRaisesRegex(RuntimeError, "LLM-keywords strategy requires"):
            AGPPipeline(self.graph).run(
                self.question, strategy="llm-keywords", generate_answer=False
            )

    def test_cli_accepts_strategy(self):
        with patch(
            "sys.argv",
            ["agp", "ask", self.question, "--strategy", "llm-keywords", "--no-answer"],
        ):
            self.assertEqual(parser().parse_args().strategy, "llm-keywords")


if __name__ == "__main__":
    unittest.main()
