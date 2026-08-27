import unittest

from agp_research.evaluation import retrieval_metrics


class EvaluationTest(unittest.TestCase):
    def test_retrieval_metrics(self):
        metrics = retrieval_metrics(["a", "b", "c"], ["b", "d"])
        self.assertEqual(metrics.precision, 1 / 3)
        self.assertEqual(metrics.recall, 1 / 2)
        self.assertEqual(metrics.hit_rate, 1.0)
        self.assertEqual(metrics.reciprocal_rank, 1 / 2)
