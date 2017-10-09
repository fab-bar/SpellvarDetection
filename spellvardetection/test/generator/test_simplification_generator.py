import unittest
import collections

from spellvardetection.generator import SimplificationGenerator

class TestSimplificationGenerator(unittest.TestCase):

    def test_getCandidates(self):

        dict = {'iu', 'ju', 'yu', 'iju', 'tu', 'hiju'}
        rules = [
            ("gh", "g"),
            ("j", "i"),
            ("v", "u"),
            ("y", "i"),
            ("th", "t"),
            ("ij", "i")
        ]
        generator = SimplificationGenerator(rules, dict)
        self.assertEqual(generator.getCandidatesForWords(['iu', 'tu']), {'iu': set(['ju', 'yu', 'iju']), 'tu': set()})

        generator = SimplificationGenerator(rules, dict, ('levenshtein', {'max_dist': 1}))
        self.assertEqual(generator.getCandidatesForWords(['iu', 'tu']), {'iu': set(['ju', 'yu', 'iju', 'hiju', 'tu']), 'tu': set(['iu', 'ju', 'yu', 'iju'])})
