import unittest

from spellvardetection.lib.lev_aut import DictAutomaton

class TestLevenshteinAutomaton(unittest.TestCase):

    def _get_matches(self, word, threshold, dictionary):

        dict_automaton = DictAutomaton(dictionary)
        return dict_automaton.fuzzySearch(word, threshold)

    def test_distance(self):

        test_dict = sorted(['Test', 'Tst', 'Tset', 'Tsset', 'abc', 'Teest', 'Teeesst'])
        matches = set(['Teest', 'Test', 'Tst'])

        self.assertEquals(
            self._get_matches('Test', 1, test_dict),
            matches)

        test_dict = sorted(['andere', 'ander'])
        matches = set(['andere', 'ander'])

        self.assertEquals(
            self._get_matches('anders', 1, test_dict),
            matches)
