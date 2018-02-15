import unittest
import json

from spellvardetection.type_filter import SKLearnClassifierBasedTypeFilter, SurfaceExtractor
from sklearn.dummy import DummyClassifier

class TestSKLearnClassifierBasedTypeFilter(unittest.TestCase):

    def setUp(self):

        self.word = 'something'
        self.candidates = set(['1', '2', '3'])

    def test_filter_canidates_none(self):

        clf = DummyClassifier(strategy='constant', constant=1)
        clf.fit([('a', 'b')], [1])
        filter = SKLearnClassifierBasedTypeFilter(clf)
        self.assertEquals(filter.filterCandidates(self.word, self.candidates), self.candidates)


    def test_filter_canidates_all(self):

        clf = DummyClassifier(strategy='constant', constant=0)
        clf.fit([('a', 'b')], [0])
        filter = SKLearnClassifierBasedTypeFilter(clf)
        self.assertEquals(filter.filterCandidates(self.word, self.candidates), set([]))
