import unittest

import spellvardetection.test.MockClasses as MockClasses

from spellvardetection.type_filter import ClusterTypeFilter

class TestClusterBasedTypeFilter(unittest.TestCase):

    def setUp(self):

        self.cluster_mock = MockClasses.Cluster({'cat', 'dog', 'hat'}, {('cat', 'dog')})

    def test_filter_candidates(self):

        filter_ = ClusterTypeFilter(self.cluster_mock)
        self.assertEquals(filter_.filterCandidates('cat', {'dog', 'hat'}),
                          {'dog'})

    def test_filter_candidates_for_word_without_cluster(self):

        filter_ = ClusterTypeFilter(self.cluster_mock)
        self.assertEquals(filter_.filterCandidates('rat', {'cat', 'dog', 'hat', 'flat'}),
                          {'cat', 'dog', 'hat', 'flat'})

    def test_keep_candidates_without_cluster(self):

        filter_ = ClusterTypeFilter(self.cluster_mock, False)
        self.assertEquals(filter_.filterCandidates('cat', {'dog', 'hat', 'flat'}),
                          {'dog', 'flat'})

    def test_remove_candidates_without_cluster(self):

        filter_ = ClusterTypeFilter(self.cluster_mock, True)
        self.assertEquals(filter_.filterCandidates('cat', {'dog', 'hat', 'flat'}),
                          {'dog'})
