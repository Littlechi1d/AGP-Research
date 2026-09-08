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
            ["Donald Trump", "Paris Agreement"],
        )

    def test_partial_word_does_not_match(self):
        self.assertEqual(
            local_keyword_extraction("Donald Trumpet", self.graph),
            [],
        )

    def test_overlapping_titles_and_separate_mentions(self):
        graph = KnowledgeGraph(
            nodes={
                "n1": Node("n1", "NASA", ""),
                "n2": Node("n2", "NASA Student Launch", ""),
            },
            edges=[],
        )
        cases = [
            ("Which pages neighbor NASA Student Launch?", ["NASA Student Launch"]),
            ("NASA and NASA Student Launch", ["NASA", "NASA Student Launch"]),
            ("NASA Student Launch and NASA", ["NASA Student Launch", "NASA"]),
            ("NASA, NASA!", ["NASA", "NASA"]),
            ("  nasa\tSTUDENT  launch?", ["NASA Student Launch"]),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(local_keyword_extraction(question, graph), expected)

    def test_press_boundaries(self):
        graph = KnowledgeGraph(nodes={"p": Node("p", "Press", "")}, edges=[])
        for question in ["Pressure", "pressing", "Express", "Press2", "_Press", "éPress"]:
            with self.subTest(question=question):
                self.assertEqual(local_keyword_extraction(question, graph), [])
        for question in ['"Press"', "(Press)", "Press,", "Press?", "Press"]:
            with self.subTest(question=question):
                self.assertEqual(local_keyword_extraction(question, graph), ["Press"])

    def test_punctuation_inside_title_is_preserved(self):
        graph = KnowledgeGraph(
            nodes={
                "n1": Node(
                    "n1",
                    "Headquarters Marine Corps, Henderson Hall, Headquarters & Service Battalion",
                    "",
                )
            },
            edges=[],
        )
        self.assertEqual(
            local_keyword_extraction(
                "Neighbor Headquarters Marine Corps, Henderson Hall, Headquarters & Service Battalion?",
                graph,
            ),
            ["Headquarters Marine Corps, Henderson Hall, Headquarters & Service Battalion"],
        )

    def test_is_is_a_fallback_stop_word(self):
        self.assertEqual(heuristic_keywords("Who is Donald Trump?"), ["Donald", "Trump"])


if __name__ == "__main__":
    unittest.main()
