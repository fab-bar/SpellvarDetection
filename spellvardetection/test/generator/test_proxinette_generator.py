import unittest

from spellvardetection.generator import ProxinetteGenerator

class TestProxinetteGenerator(unittest.TestCase):

    def test_getCandidates(self):

        generator = ProxinetteGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.08)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})
        generator = ProxinetteGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.09)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(), 'dog': set()})
