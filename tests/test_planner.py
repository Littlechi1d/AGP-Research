import unittest

from agp_research.graph import KnowledgeGraph
from agp_research.models import Node
from agp_research.planner import heuristic_keywords, local_keyword_extraction


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph(
            nodes={
                "n1": Node("n1", "Donald Trump", "A person"),
                "n2": Node("n2", "Paris Agreement", "An agreement"),
            },
            edges=[],
        )

    def test_local_keyword_before_question_mark(self):
        self.assertEqual(
            local_keyword_extraction("Who is Donald Trump?", self.graph),
            ["Donald Trump"],
        )

    def test_local_keyword_before_other_punctuation(self):
        self.assertEqual(
            local_keyword_extraction("Explain Donald Trump, then Paris Agreement.", self.graph),
            ["Paris Agreement", "Donald Trump"],
        )

    def test_partial_word_does_not_match(self):
        self.assertEqual(
            local_keyword_extraction("Donald Trumpet", self.graph),
            [],
        )

    def test_is_is_a_fallback_stop_word(self):
        self.assertEqual(heuristic_keywords("Who is Donald Trump?"), ["Donald", "Trump"])


if __name__ == "__main__":
    unittest.main()
