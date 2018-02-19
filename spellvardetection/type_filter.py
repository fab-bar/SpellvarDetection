# -*- coding: utf-8 -*-

import abc
import json
from threading import Lock

import numpy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

import spellvardetection.lib.util


class _AbstractTypeFilter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def isPair(self, word, candidate):
        pass

    def filterCandidates(self, word, candidates):

        return set([candidate for candidate in candidates if self.isPair(word, candidate)])

class SKLearnClassifierBasedTypeFilter(_AbstractTypeFilter):

    def __init__(self, classifier):

        self.classifier = classifier

    def isPair(self, word, candidate):

        if self.classifier.predict([(word, candidate)]) == 1:
            return True
        else:
            return False

class FeatureExtractorMixin(metaclass=abc.ABCMeta):

    lock = Lock()


    @abc.abstractmethod
    def _featureExtraction(self, datapoint):
        pass

    def extractFeaturesFromDatapoint(self, datapoint):

        ## query cache
        with FeatureExtractorMixin.lock:
            if hasattr(self, 'feature_cache'):
                if json.dumps(tuple(sorted(datapoint))) in self.feature_cache:
                    return self.feature_cache[json.dumps(tuple(sorted(datapoint)))]

        features = self._featureExtraction(datapoint)

        ## update cache
        with FeatureExtractorMixin.lock:
            if hasattr(self, 'feature_cache'):

                self.feature_cache[json.dumps(tuple(sorted(datapoint)))] = features

        return features

    def extractFeatures(self, data):

        return [self.extractFeaturesFromDatapoint(datapoint) for datapoint in data]

    def setFeatureCache(self, feature_cache=None):

        if feature_cache is None:
            self.feature_cache = dict()
        else:
            self.feature_cache = spellvardetection.lib.util.load_from_file_if_string(feature_cache)


class SurfaceExtractor(BaseEstimator, TransformerMixin, FeatureExtractorMixin):

    def __init__(self, min_ngram_size=1, max_ngram_size=3, only_mismatch_ngrams=True, padding_char="$"):

        self.padding_char = padding_char
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.only_mismatch_ngrams = only_mismatch_ngrams

    # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    def _find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])


    def _featureExtraction(self, data_point):

        word = data_point[0]
        candidate = data_point[1]

        ## align word and candidate
        alignment = list(spellvardetection.lib.util.get_alignment(self.padding_char + word + self.padding_char,
                                             self.padding_char + candidate + self.padding_char))
        ## get ngrams from alignment (size is option)
        ngrams = set()
        for i in range(self.min_ngram_size, self.max_ngram_size+1):
            ngrams.update(list(self._find_ngrams(alignment, i)))
        ngrams = list(ngrams)

        ## only alignment from mismatch
        if self.only_mismatch_ngrams:
            ngrams = list(filter(lambda x: any([len(pair) > 1 and pair[0] != pair[1] in pair for pair in x]), ngrams))

        return ngrams


    def fit(self, data, y=None):

        self.vec = DictVectorizer(dtype=numpy.int64)
        self.vec.fit(list(map(lambda ngrams: {tuple(key): 1 for key in ngrams},
                              [self.extractFeaturesFromDatapoint(observation) for observation in data])))

        return self

    def transform(self, data):

        return self.vec.transform(list(map(lambda ngrams: {tuple(key): 1 for key in ngrams},
                                           [self.extractFeaturesFromDatapoint(observation) for observation in data])))
