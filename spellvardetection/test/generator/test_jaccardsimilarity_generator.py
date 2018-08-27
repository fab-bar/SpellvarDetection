import unittest

from spellvardetection.generator import JaccardSimilarityGenerator, FrequencyWeightedJaccardSimilarityGenerator

class TestJaccardSimilarityGenerator(unittest.TestCase):

    def test_getCandidates(self):

        generator = JaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.2)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})
        generator = JaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.25)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat']), 'dog': set()})
        generator = JaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.8)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(), 'dog': set()})

class TestFrequencyWeightedJaccardSimilarityGenerator(unittest.TestCase):

    def test_getCandidates(self):

        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.08)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})
        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.1)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat']), 'dog': set()})
        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.11)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(), 'dog': set()})
