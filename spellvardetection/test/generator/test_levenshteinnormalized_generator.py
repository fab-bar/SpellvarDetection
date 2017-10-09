import unittest

from spellvardetection.generator import LevenshteinNormalizedGenerator
from spellvardetection.generator import LevenshteinGenerator

class TestLevenshteinNormalizedGenerator(unittest.TestCase):

    def test_getCandidates(self):

        test_dict = ['cat', 'mat', 'hat', 'dog' 'apple', 'flat']

        generator_lev_0 = LevenshteinGenerator(test_dict, 0)
        generator_lev_1 = LevenshteinGenerator(test_dict, 1)
        generator_lev_2 = LevenshteinGenerator(test_dict, 2)

        # Levenshtein distance 0
        target_set = generator_lev_0.getCandidatesForWords(['rat', 'dog'])
        generator = LevenshteinNormalizedGenerator(['cat', 'mat', 'hat', 'dog' 'apple', 'flat'], 0.1, False)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), target_set)

        # Levenshtein distance 1
        target_set = generator_lev_1.getCandidatesForWords(['rat', 'dog'])
        generator = LevenshteinNormalizedGenerator(['cat', 'mat', 'hat', 'dog' 'apple', 'flat'], 0.1, True)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), target_set)

        generator = LevenshteinNormalizedGenerator(['cat', 'mat', 'hat', 'dog' 'apple', 'flat'], 0.34, False)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), target_set)

        # Levenshtein distance 2
        target_set = generator_lev_2.getCandidatesForWords(['rat', 'dog'])
        generator = LevenshteinNormalizedGenerator(['cat', 'mat', 'hat', 'dog' 'apple', 'flat'], 0.7, False)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), target_set)

        generator = LevenshteinNormalizedGenerator(['cat', 'mat', 'hat', 'dog' 'apple', 'flat'], 0.7, True)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), target_set)

        # different distances
        generator = LevenshteinNormalizedGenerator(['cat', 'mat', 'hat', 'dog' 'apple', 'flat'], 0.5, False)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'coat']),
                         {
                             **generator_lev_1.getCandidatesForWords(['rat']),
                             **generator_lev_2.getCandidatesForWords(['coat'])
                         }
        )
