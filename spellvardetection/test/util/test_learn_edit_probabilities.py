import unittest

import spellvardetection.util.learn_edit_probabilities as probs

class TestLearnEditProbabilities(unittest.TestCase):

    def test_getProbabilitiesFromSpellvars(self):

        self.assertEqual(
            probs.getProbabilitiesFromSpellvars({'aa': ['ac'], 'ac': []}),
            [{'char1': 'a', 'char2': 'c', 'probability': 2/3}]
        )
