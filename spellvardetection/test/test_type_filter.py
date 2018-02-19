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

class TestSurfaceExtractor(unittest.TestCase):

    def setUp(self):

        self.data_point = ('test', 'fest')

    ## test cache of the mixin
    def test_feature_extractor_cache(self):

        ext = SurfaceExtractor()

        ## test cache hit
        ext.setFeatureCache({
            json.dumps(tuple(sorted(self.data_point))): 'a'
        })

        self.assertEquals(
            ext.extractFeaturesFromDatapoint(self.data_point),
            'a'
        )

        ## test empy cache
        ext.setFeatureCache()
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])
        )


    def test_extract_all_ngrams(self):

        ext = SurfaceExtractor(only_mismatch_ngrams=False)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([
                ('$$',), ('ft',), ('ee',), ('ss',), ('tt',),
                ('$$', 'ft'), ('ft', 'ee'), ('ee', 'ss'), ('ss', 'tt'), ('tt', '$$'),
                ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss'), ('ee', 'ss', 'tt'), ('ss', 'tt', '$$')
            ])
        )

    def test_extract_mismatch_ngrams(self):

        ext = SurfaceExtractor(only_mismatch_ngrams=True)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',), ('$$', 'ft'), ('ft', 'ee'), ('$$', 'ft', 'ee'), ('ft', 'ee', 'ss')])
        )


    def test_extract_all_unigrams(self):

        ext = SurfaceExtractor(max_ngram_size=1)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('ft',)])
        )


    def test_extract_all_bigrams(self):

        ext = SurfaceExtractor(min_ngram_size=2, max_ngram_size=2)
        self.assertEquals(
            set(ext.extractFeaturesFromDatapoint(self.data_point)),
            set([('$$', 'ft'), ('ft', 'ee')])
        )
