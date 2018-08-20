import unittest

from spellvardetection.generator import LevenshteinGenerator

class TestLevenshteinGenerator(unittest.TestCase):

    def test_getCandidates(self):

        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 1)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat']), 'dog': set()})
        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 2)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})
