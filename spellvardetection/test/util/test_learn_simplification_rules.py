import unittest

import spellvardetection.util.learn_simplification_rules as simpl

class TestLearnSimplificationRules(unittest.TestCase):

    def test_getRules_substitution(self):

        self.assertEqual(simpl.getRules("ein", "eyn")[0]['rule'],
                         ("i", "y"))

        self.assertEqual(simpl.getRules("eyn", "ein")[0]['rule'],
                         ("i", "y"))

    def test_getRules_deletion(self):
        self.assertEqual(simpl.getRules("ghe", "ge")[0]['rule'],
                         ("gh", "g"))

        self.assertEqual(simpl.getRules("ge", "ghe")[0]['rule'],
                         ("gh", "g"))

        ## padding is included
        self.assertEqual(simpl.getRules("ge", "e")[0]['rule'],
                         ("^g", "^"))

    def test_getRules_mergesplit(self):

        self.assertEqual(simpl.getRules("th", "f")[0]['rule'],
                         ('th', 'f'))

    def test_getRules_context(self):

        self.assertEqual(simpl.getRules("ein", "eyn")[0]['left_context'],
                         "e")
        self.assertEqual(simpl.getRules("ein", "eyn")[0]['right_context'],
                         "n")


    def test_getPairsFromSpellvardict(self):

        self.assertEqual(
            simpl.getPairsFromSpellvardict({'abc': ['a', 'b', 'c'], 'c': ['a', 'abc']}),
            set([('a', 'abc'), ('a', 'c'), ('abc', 'b'), ('abc', 'c')])
        )

    def test_getRulesAndFreqFromSpellvars(self):

        self.assertEqual(
            simpl.getRulesAndFreqFromSpellvars({'aa': ['ac'], 'ac': []}),
            {('a', 'c'): {'tp': 1, 'fp': 0}}
        )


        self.assertEqual(
            simpl.getRulesAndFreqFromSpellvars({'aad': ['cad'], 'aat': [], 'cat': []}),
            {('a', 'c'): {'tp': 1, 'fp': 1}}
        )
