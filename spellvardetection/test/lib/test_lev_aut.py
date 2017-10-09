import unittest

from spellvardetection.lib.lev_aut import DictAutomaton

class TestLevenshteinAutomaton(unittest.TestCase):

    def _get_matches(self, word, threshold, dictionary, merge_split=False, transposition=False, repetitions=False):

        dict_automaton = DictAutomaton(dictionary)
        return dict_automaton.fuzzySearch(word, threshold, merge_split, transposition, repetitions)

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

    def test_split_merge(self):

        test_dict = ['water', 'uuasser']
        self.assertEquals(
            self._get_matches('wasser', 1, test_dict),
            set([]))
        self.assertEquals(
            self._get_matches('wasser', 1, test_dict, merge_split=True),
            set(['uuasser', 'water']))

    def test_transposition(self):

        test_dict = ['Test', 'Tst', 'Tset', 'Tsset', 'abc', 'Teest', 'Teeesst']
        matches = set(['Teest', 'Test', 'Tset', 'Tst'])

        self.assertEquals(
            self._get_matches('Test', 1, test_dict, transposition=True),
            matches)

        ## it is not the Damerau-Levenshtein distance
        test_dict = ['ABC']
        self.assertEquals(
            self._get_matches('CA', 2, test_dict),
            set([]))
        self.assertEquals(
            self._get_matches('CA', 3, test_dict),
            set(['ABC'])
        )

    def test_repetitions(self):

        self.assertEquals(
            self._get_matches('Test', 1, ['Test', 'Tst', 'Tset', 'Tsset', 'abc', 'Teest', 'Teeesst'], repetitions=True),
            set(['Teeesst', 'Teest', 'Test', 'Tst']))

        ## Repetition the other way round
        self.assertEquals(
            self._get_matches('loool', 1, ['loooooool', 'lol'], repetitions=True),
            set(['lol', 'loooooool']))

        ## Repetitions are only valid if the repeated character is shared
        self.assertEquals(
            self._get_matches('difffffffff', 1, ['diff', 'diy'], repetitions=True),
            set(['diff']))

    def test_transpostion_and_repetition(self):
        self.assertEquals(
            self._get_matches('Test', 1, ['Test', 'Tst', 'Tset', 'Tsset', 'abc', 'Teest', 'Teeesst'], transposition=True, repetitions=True),
            set(['Test', 'Tst', 'Tset', 'Teest', 'Teeesst']))
        self.assertEquals(
            self._get_matches('Test', 1, ['Test', 'Tehst', 'Tast', 'Tst', 'Tset', 'Tsset', 'abc', 'Teest', 'Teeesst'], transposition=True, repetitions=True),
            set(['Tast', 'Teeesst', 'Teest', 'Tehst', 'Test', 'Tset', 'Tst']))

