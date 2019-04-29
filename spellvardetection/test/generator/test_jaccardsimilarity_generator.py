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

    def test_getCandidates_with_similarity(self):

        generator = JaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.2, add_similarity=True)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set([('cat', 0.25), ('mat', 0.25), ('hat', 0.25), ('flat', 0.2)]), 'dog': set()})

class TestFrequencyWeightedJaccardSimilarityGenerator(unittest.TestCase):

    def test_getCandidates(self):

        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.08)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})
        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.1)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat']), 'dog': set()})
        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.11)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(), 'dog': set()})

    def test_getCandidates_with_similarity(self):

        generator = FrequencyWeightedJaccardSimilarityGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.08, add_similarity=True)
        candidates = generator.getCandidatesForWords(['rat', 'dog'])
        self.assertEqual(candidates['dog'], set())
        expected_results = {
            'cat': 0.108108108,
            'mat': 0.108108108,
            'hat': 0.108108108,
            'flat': 0.08510638
        }
        # assert that the expected candidates are returned
        self.assertEquals(set(map(lambda candidate: candidate[0], candidates['rat'])), set(expected_results.keys()))
        # assert that the similarity values for candidates are as expected
        for candidate in candidates['rat']:
            self.assertAlmostEqual(candidate[1], expected_results[candidate[0]])
