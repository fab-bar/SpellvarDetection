import unittest

from spellvardetection.type_filter import EditProbabilitiesTypeFilter

class TestEditProbabilityTypeFilter(unittest.TestCase):

    def test_filter_candidates(self):

        filter_ = EditProbabilitiesTypeFilter(
            [
                {'char1': 'u', 'char2': 'v', 'probability': 1},
                {'char1': 'e', 'char2': '', 'probability': 0.9}
            ],
            1
        )
        self.assertEquals(filter_.filterCandidates('unde', {'und', 'vnd', 'vnde', 'vnder'}),
                          {'vnde', 'vnder'})

    def test_not_all_weights_are_probabilities(self):

        with self.assertRaises(ValueError):
            filter_ = EditProbabilitiesTypeFilter(
                [
                    {'char1': 'u', 'char2': 'v', 'probability': 2},
                ],
                1
            )


    def test_zero_probabilities(self):

        with self.assertRaises(ValueError):
            filter_ = EditProbabilitiesTypeFilter(
                [
                    {'char1': 'u', 'char2': 'v', 'probability': 0},
                ],
                1
            )
    def test_duplicates(self):

        with self.assertRaises(ValueError):
            filter_ = EditProbabilitiesTypeFilter(
                [
                    {'char1': 'u', 'char2': 'v', 'probability': 0},
                    {'char1': 'v', 'char2': 'u', 'probability': 1},
                ],
                1
            )
