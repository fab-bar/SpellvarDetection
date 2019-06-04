import unittest

from spellvardetection.generator import LevenshteinGenerator

class TestLevenshteinGenerator(unittest.TestCase):

    def test_without_dictionary(self):

        generator = LevenshteinGenerator()
        with self.assertRaises(RuntimeError):
            generator.getCandidatesForWords(['rat', 'dog'])

    def test_getCandidates(self):

        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 1)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat']), 'dog': set()})
        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 2)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})

    def test_getCandidates_strict(self):

        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 1, strict_dist=True)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat']), 'dog': set()})
        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 2, strict_dist=True)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['flat']), 'dog': set()})

    def test_set_dictionary(self):
        generator = LevenshteinGenerator(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 1)
        generator.setDictionary(['cat', 'mat', 'dog', 'apple', 'flat'])
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat']), 'dog': set()})
