import unittest

from agp_research.mapping_evaluation import select_configuration


class MappingEvaluationTest(unittest.TestCase):
    def test_selection_prioritizes_precision_and_then_conservative_controls(self):
        results = [
            {"precision": .9, "f1": 1.0, "recall": 1.0, "threshold": .7, "margin": 0},
            {"precision": 1.0, "f1": .95, "recall": .9, "threshold": .8, "margin": .05},
            {"precision": 1.0, "f1": .95, "recall": .9, "threshold": .85, "margin": .1},
        ]

        selected = select_configuration(results)

        self.assertEqual(selected["threshold"], .85)
        self.assertEqual(selected["margin"], .1)

    def test_selection_requires_results(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            select_configuration([])


if __name__ == "__main__":
    unittest.main()
