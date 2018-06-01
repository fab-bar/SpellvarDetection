import unittest

from spellvardetection.lib.util import *

from spellvardetection.generator import LevenshteinGenerator

class TestAlignment(unittest.TestCase):

    def test_alignment_basic_sequences(self):

        self.assertEquals(
            list(get_alignment('AGT', 'ABGGTGTG')),
            ['AA', '-B', 'GG', '-G', 'TT', '-G', '-T', '-G']
        )
        self.assertEquals(
            list(get_alignment('CAGACGT', 'CGATA')),
            ['CC', '-A', 'GG', 'AA', '-C', '-G', 'TT', '-A']
        )
        self.assertEquals(
            list(get_alignment('12345678', '123478901')),
            ['11', '22', '33', '44', '-5', '-6', '77', '88', '-9', '-0', '-1']
        )
        self.assertEquals(
            list(get_alignment('45678', '123478901')),
            ['14', '25', '36', '-4', '77', '88', '-9', '-0', '-1']
        )

    def test_alignment_at_end(self):

        self.assertEquals(
            list(get_alignment('test', 'est')),
            ['-t', 'ee', 'ss', 'tt']
        )

        self.assertEquals(
            list(get_alignment('est', 'test')),
            ['-t', 'ee', 'ss', 'tt']
        )

    def test_alignment_order(self):

        self.assertEquals(
            list(get_alignment('dit', 'dyt')),
            list(get_alignment('dyt', 'dit')),
        )

        self.assertEquals(
            list(get_alignment('12345678', '478901')),
            list(get_alignment('478901', '12345678'))
        )

    def test_conflate_id(self):

        self.assertEquals(
            list(get_alignment('dit', 'dyt', conflate_id_pairs=True)),
            ['IDD', 'iy', 'IDD']
        )

class TestEvaluate(unittest.TestCase):


    def test_evaluate_perfect_prediction(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': ['dit']},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': ['ist']}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, [1, 1, 1, 1, 0])


    def test_evaluate_perfect_recall(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': ['dit', 'ist']},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': ['dit', 'ist']}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, [0.5, 1, 0.6666666666666666, 2, 0])


    def test_evaluate_empty_predictions(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': []},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': []}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, [1, 0, 0, 0, 0])


    def test_evaluate_only_false_predictions(self):
        tokens = [
            {'type': 'dyt', 'variants': ['dit'], 'filtered_candidates': ['ist']},
            {'type': 'is', 'variants': ['ist'], 'filtered_candidates': ['dyt']}
        ]

        result = evaluate(tokens)

        self.assertEquals(result, [0, 0, 0, 1, 0])

    def test_getPairsFromSpellvardict(self):

        self.assertEqual(
            getPairsFromSpellvardict({'abc': ['a', 'b', 'c'], 'c': ['a', 'abc']}),
            set([('a', 'abc'), ('a', 'c'), ('abc', 'b'), ('abc', 'c')])
        )

    def test_getTrueAndFalsePairs(self):

        generator = LevenshteinGenerator(max_dist=1)
        true_pairs, false_pairs = getTrueAndFalsePairs({'abc': ['a', 'b', 'c'], 'c': ['a', 'abc']}, generator)
        self.assertEqual(
            true_pairs,
            set([('a', 'c')])
        )
        self.assertEqual(
            false_pairs,
            set([('a', 'b'), ('b', 'c')])
        )


class TestPairsWithContext(unittest.TestCase):

    def test_get_positive_and_negative_pairs_with_context(self):


        tokens = [
            { 'type': 'a', 'variants': [], 'left_context': [], 'right_context': ['short', 'example', 'for', 'testing']},
            { 'type': 'short', 'variants': ['shoort'], 'left_context': ['a'], 'right_context': ['example', 'for', 'testing']},
            { 'type': 'example', 'variants': ['exaample', 'exmple'], 'left_context': ['a', 'short'], 'right_context': ['for', 'testing']},
            { 'type': 'for', 'variants': ['4'], 'left_context': ['a', 'short', 'example'], 'right_context': ['testing']},
            { 'type': 'testing', 'variants': ['tstin'], 'left_context': ['a', 'short', 'example', 'for'], 'right_context': []},
        ]

        spellvardict = {
            'short': ['shoort', 'shrt'],
            'example': ['exaample', 'exmple'],
        }

        positive_pairs, negative_pairs = get_positive_and_negative_pairs_with_context(tokens, spellvardict)

        ## numbers of examples
        self.assertEquals(3, len(positive_pairs))
        self.assertEquals(1, len(negative_pairs))
