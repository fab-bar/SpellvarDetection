import unittest

from spellvardetection.generator import ProxinetteGenerator

class TestProxinetteGenerator(unittest.TestCase):

    def test_without_dictionary(self):

        generator = ProxinetteGenerator.create()
        with self.assertRaises(RuntimeError):
            generator.getCandidatesForWords(['rat', 'dog'])

    def test_getCandidates(self):

        generator = ProxinetteGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.08)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat', 'hat', 'flat']), 'dog': set()})
        generator = ProxinetteGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.09)
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(), 'dog': set()})

    def test_set_dictionary(self):
        generator = ProxinetteGenerator.create(['cat', 'mat', 'hat', 'dog', 'apple', 'flat'], 0.08)
        generator.setDictionary(['cat', 'mat', 'dog', 'apple'])
        self.assertEqual(generator.getCandidatesForWords(['rat', 'dog']), {'rat': set(['cat', 'mat']), 'dog': set()})
