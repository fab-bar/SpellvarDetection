import unittest

from spellvardetection.type_filter import SKLearnClassifierBasedTypeFilter
from spellvardetection.util.feature_extractor import SurfaceExtractor
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from imblearn.ensemble import BalancedBaggingClassifier

class TestSKLearnClassifierBasedTypeFilter(unittest.TestCase):

    def setUp(self):

        self.word = 'something'
        self.candidates = set(['1', '2', '3'])

    def test_filter_canidates_none(self):

        clf = DummyClassifier(strategy='constant', constant=1)
        filter = SKLearnClassifierBasedTypeFilter(clf)
        filter.fit([('a', 'b')], [1])
        self.assertEquals(filter.filterCandidates(self.word, self.candidates), self.candidates)


    def test_filter_canidates_all(self):

        clf = DummyClassifier(strategy='constant', constant=0)
        filter = SKLearnClassifierBasedTypeFilter(clf)
        filter.fit([('a', 'b')], [0])
        self.assertEquals(filter.filterCandidates(self.word, self.candidates), set([]))

    def test_create_with_qualified_classname(self):
        clf = SKLearnClassifierBasedTypeFilter.create('sklearn.dummy.DummyClassifier', [])
        self.assertTrue(isinstance(clf.classifier, DummyClassifier))

    def test_create_from_string_svm(self):
        clf = SKLearnClassifierBasedTypeFilter.create('__svm__', [])
        self.assertTrue(isinstance(clf.classifier, SVC))

    def test_create_from_string_bsvm(self):
        clf = SKLearnClassifierBasedTypeFilter.create('__bagging_svm__', [])
        self.assertTrue(isinstance(clf.classifier, BalancedBaggingClassifier))

    def test_create_with_classifier_params(self):
        clf = SKLearnClassifierBasedTypeFilter.create('SVC', [], {'C': 2, 'gamma': 0.1})
        params = clf.get_params()
        self.assertEquals(params['classifier__C'], 2)
        self.assertEquals(params['classifier__gamma'], 0.1)

    def test_create_with_feature_extractor(self):
        clf = SKLearnClassifierBasedTypeFilter.create('__svm__', [
            {'name': 'Feat1', 'type': 'surface',
             'options': {'padding_char': 'Test'}}])

        extractor_name, extractor = clf.feature_extractors[0]
        self.assertEquals(extractor_name, 'Feat1')
        self.assertTrue(isinstance(extractor, SurfaceExtractor))
        self.assertEquals(extractor.padding_char, 'Test')

