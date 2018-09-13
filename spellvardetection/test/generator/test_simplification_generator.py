import unittest
import collections

from spellvardetection.generator import SimplificationGenerator

class TestSimplificationGenerator(unittest.TestCase):

    rules = [
        ("gh", "g"),
        ("j", "i"),
        ("v", "u"),
        ("y", "i"),
        ("th", "t"),
        ("ij", "i")
    ]
    dict = {'iu', 'ju', 'yu', 'iju', 'tu', 'hiju'}

    def test_without_dictionary(self):

        generator = SimplificationGenerator(self.rules)
        with self.assertRaises(RuntimeError):
            generator.getCandidatesForWords(['rat', 'dog'])

    def test_getCandidates(self):

        generator = SimplificationGenerator(self.rules, self.dict)
        self.assertEqual(generator.getCandidatesForWords(['iu', 'tu']), {'iu': set(['ju', 'yu', 'iju']), 'tu': set()})

        generator = SimplificationGenerator(self.rules, self.dict, {'type': 'levenshtein', 'options': {'max_dist': 1}})
        self.assertEqual(generator.getCandidatesForWords(['iu', 'tu']), {'iu': set(['ju', 'yu', 'iju', 'hiju', 'tu']), 'tu': set(['iu', 'ju', 'yu', 'iju'])})

    def test_set_dictionary(self):

        generator = SimplificationGenerator(self.rules, self.dict)
        generator.setDictionary({'iu', 'ju', 'yu', 'tu', 'hiju'})
        self.assertEqual(generator.getCandidatesForWords(['iu', 'tu']), {'iu': set(['ju', 'yu']), 'tu': set()})
