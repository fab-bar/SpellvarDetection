import unittest

from spellvardetection.lib.util import *

class TestEvaluate(unittest.TestCase):


    def test_evaluate_perfect_prediction(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': ['dit']},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': ['ist']}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, '1.00|1.00|1.00|1.00+-0.00')


    def test_evaluate_perfect_recall(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': ['dit', 'ist']},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': ['dit', 'ist']}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, '0.50|1.00|0.67|2.00+-0.00')


    def test_evaluate_empty_predictions(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': []},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': []}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, '1.00|0.00|0.00|0.00+-0.00')


    def test_evaluate_only_false_predictions(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': ['ist']},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': ['dyt']}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, '0.00|0.00|0.00|1.00+-0.00')
