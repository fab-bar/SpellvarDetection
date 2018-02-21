# -*- coding: utf-8 -*-

import abc
import json
from threading import Lock

import numpy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


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
                    features = self.feature_cache[json.dumps(tuple(sorted(datapoint)))]
                    if self.key is not None:
                        features = features.get(self.key, None)
                    if features is not None:
                        return features

        features = self._featureExtraction(datapoint)

        ## update cache
        with FeatureExtractorMixin.lock:
            if hasattr(self, 'feature_cache'):

                if self.key is None:
                    self.feature_cache[json.dumps(tuple(sorted(datapoint)))] = features
                else:
                    if json.dumps(tuple(sorted(datapoint))) not in self.feature_cache:
                        self.feature_cache[json.dumps(tuple(sorted(datapoint)))] = dict()
                    self.feature_cache[json.dumps(tuple(sorted(datapoint)))][self.key] = features

        return features

    def extractFeatures(self, data):

        return [self.extractFeaturesFromDatapoint(datapoint) for datapoint in data]

    def setFeatureCache(self, feature_cache=None, key=None):

        if feature_cache is None:
            self.feature_cache = dict()
        else:
            self.feature_cache = spellvardetection.lib.util.load_from_file_if_string(feature_cache)

        self.key = key


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

class ContextExtractor(BaseEstimator, TransformerMixin, FeatureExtractorMixin):

    def __init__(self, embeddings):

        self.embeddings = embeddings

    def _featureExtraction(self, data_point):

        word = data_point[0]
        candidate = data_point[1]

        word_embedd = self.embeddings.get(word)
        cand_embedd = self.embeddings.get(candidate)
        if not (word_embedd is None or cand_embedd is None):
            features = cosine_similarity([word_embedd], [cand_embedd])[0][0]
        else:
            features = float('nan')

        return features

    def fit(self, data, y=None):

        self.preprocessing = Pipeline([
            ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
            ('normalizer', StandardScaler())
        ])
        self.preprocessing.fit(numpy.array([self.extractFeaturesFromDatapoint(x) for x in data]).reshape(-1, 1))

        return self

    def transform(self, data):

        return self.preprocessing.transform(numpy.array([self.extractFeaturesFromDatapoint(x) for x in data]).reshape(-1, 1))
