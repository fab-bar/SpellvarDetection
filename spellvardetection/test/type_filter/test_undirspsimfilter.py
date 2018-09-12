import unittest

from spellvardetection.type_filter import UndirSpSimTypeFilter


class TestUndirSpSimTypeFilter(unittest.TestCase):

    def test_filter_candidates_without_training(self):

        filter_ = UndirSpSimTypeFilter.create_for_training()
        self.assertEquals(filter_.filterCandidates('phase', {'fase', 'hase'}),
                          set())

    def test_filter_with_training(self):

        filter_ = UndirSpSimTypeFilter.create_for_training()
        filter_.train([('phase', 'fase')], [])
        self.assertEquals(filter_.filterCandidates('phase', {'fase', 'hase'}),
                          {'fase'})
