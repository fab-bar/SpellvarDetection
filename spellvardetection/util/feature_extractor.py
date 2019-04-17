 # -*- coding: utf-8 -*-

import abc
import inspect
import itertools
import json
from threading import Lock

import numpy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

import spellvardetection.lib.embeddings
import spellvardetection.lib.util


class FeatureExtractorMixin(metaclass=abc.ABCMeta):

    lock = Lock()

    ## prevent cache from being pickled
    ## subclasses can provide __getstate__ to override this behaviour
    def __getstate__(self):

        pickle_dict = dict(self.__dict__)
        if 'feature_cache' in pickle_dict:
            del pickle_dict['feature_cache']
        return pickle_dict

    def __init_subclass__(cls, **kwargs):

        if not hasattr(cls, '__getstate__'):
            setattr(cls, '__getstate__', FeatureExtractorMixin.__getstate__)

        ## add attributes cache and key to create function and set the cache
        ## subclasses create method can add cache or key as parameter to the create method to override this behaviour
        if hasattr(cls, 'create'):

            func = getattr(cls, 'create')

            sig = inspect.signature(getattr(cls, 'create'))
            if "cache" not in sig.parameters and "key" not in sig.parameters:

                def create_and_add_cache(*args, cache: dict=None, key=None, **kwargs):

                    extractor = func(*args, **kwargs)
                    if cache is not None:
                        extractor.setFeatureCache(cache, key)
                    return extractor

                new_signature = list(sig.parameters.values()) + [
                    inspect.Parameter("cache", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None, annotation=dict),
                    inspect.Parameter("key", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)]

                create_and_add_cache.__signature__ = inspect.Signature(new_signature)

                setattr(cls, 'create', create_and_add_cache)

    @abc.abstractmethod
    def create(**options):
        pass

    def _getDataKey(self, datapoint):
        return json.dumps(tuple(sorted(datapoint)))

    @abc.abstractmethod
    def _featureExtraction(self, datapoint):
        pass

    def extractFeaturesFromDatapoint(self, datapoint):

        ## query cache
        with FeatureExtractorMixin.lock:
            if hasattr(self, 'feature_cache'):
                if self._getDataKey(datapoint) in self.feature_cache:
                    features = self.feature_cache[self._getDataKey(datapoint)]
                    if self.key is not None:
                        features = features.get(self.key, None)
                    if features is not None:
                        return features

        features = self._featureExtraction(datapoint)

        ## update cache
        with FeatureExtractorMixin.lock:
            if hasattr(self, 'feature_cache'):

                if self.key is None:
                    self.feature_cache[self._getDataKey(datapoint)] = features
                else:
                    if self._getDataKey(datapoint) not in self.feature_cache:
                        self.feature_cache[self._getDataKey(datapoint)] = dict()
                    self.feature_cache[self._getDataKey(datapoint)][self.key] = features

        return features

    def extractFeatures(self, data):

        return [self.extractFeaturesFromDatapoint(datapoint) for datapoint in data]

    def setFeatureCache(self, feature_cache=None, key=None):

        if feature_cache is None:
            self.feature_cache = dict()
        else:
            self.feature_cache = feature_cache

        self.key = key

class NGramExtractor(FeatureExtractorMixin):

    name = "ngram"

    def create(min_ngram_size=2, max_ngram_size=float('inf'), skip_size=0, gap="|", bow="$", eow="$", pad_ngrams=False):
        return NGramExtractor(min_ngram_size, max_ngram_size, skip_size, gap, bow, eow)

    def __init__(self, min_ngram_size=2, max_ngram_size=float('inf'), skip_size=0, gap="|", bow="$", eow="$", pad_ngrams=False):

        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.skip_size = skip_size
        self.gap = gap
        self.bow = bow
        self.eow = eow
        self.pad_ngrams = pad_ngrams

    # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    def _find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def _featureExtraction(self, datapoint):

        padded_datapoint = list(datapoint)
        if self.bow or self.eow:
            if self.bow:
                if self.pad_ngrams:
                    padded_datapoint[0] = self.bow + padded_datapoint[0]
                else:
                    padded_datapoint = [self.bow] + padded_datapoint
            if self.eow:
                if self.pad_ngrams:
                    padded_datapoint[-1] = padded_datapoint[-1] + self.eow
                else:
                    padded_datapoint = padded_datapoint + [self.eow]

        ngrams = set()
        for i in range(self.min_ngram_size, min(len(datapoint),self.max_ngram_size)+1):
            ngrams.update(list(self._find_ngrams(padded_datapoint, i)))
            ### add skip grams for the given size
            for skip in range(1, min(self.skip_size, len(datapoint) - i) + 1):
                ngrams_for_skipgrams = list(self._find_ngrams(datapoint, i + skip))
                ### get all combinations of (i-2) times True and skip times False
                for skip_vector in set(list(itertools.permutations([False]*skip + [True]*(i-2)))):
                    skip_vector = [True] + list(skip_vector) + [True]
                    if self.gap:
                        skipgrams = [tuple([char if skip_vector[position] else self.gap for position, char in enumerate(ngram)]) for ngram in ngrams_for_skipgrams]
                    else:
                        skipgrams = [tuple([char for position, char in enumerate(ngram) if skip_vector[position]]) for ngram in ngrams_for_skipgrams]
                    ngrams.update(skipgrams)

        return ngrams

class SurfaceExtractor(BaseEstimator, TransformerMixin, FeatureExtractorMixin):

    name = "surface"

    def create(min_ngram_size=1, max_ngram_size=3, only_mismatch_ngrams=True, padding_char="$"):
        return SurfaceExtractor(min_ngram_size, max_ngram_size, only_mismatch_ngrams, padding_char)

    def __init__(self, min_ngram_size=1, max_ngram_size=3, only_mismatch_ngrams=True, padding_char="$"):

        self.ngram_extractor = NGramExtractor(min_ngram_size, max_ngram_size, skip_size=0,
                                              bow='', eow='')
        self.padding_char = padding_char
        self.only_mismatch_ngrams = only_mismatch_ngrams

    def _featureExtraction(self, data_point):

        word = data_point[0]
        candidate = data_point[1]

        ## align word and candidate
        alignment = list(spellvardetection.lib.util.get_alignment(self.padding_char + word + self.padding_char,
                                             self.padding_char + candidate + self.padding_char))
        ## get ngrams from alignment (size is option)
        ngrams = list(self.ngram_extractor.extractFeaturesFromDatapoint(alignment))

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

    name = 'context'

    def create(vector_type, vectorfile_name, simplfile_name=None, missing_words=None):

        embeddings = spellvardetection.lib.embeddings.WordEmbeddings(vector_type, vectorfile_name, simplfile_name, missing_words)
        return ContextExtractor(embeddings)

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
            ('imputer', SimpleImputer(strategy='mean')),
            ('normalizer', StandardScaler())
        ])
        self.preprocessing.fit(numpy.array([self.extractFeaturesFromDatapoint(x) for x in data]).reshape(-1, 1))

        return self

    def transform(self, data):

        return self.preprocessing.transform(numpy.array([self.extractFeaturesFromDatapoint(x) for x in data]).reshape(-1, 1))
