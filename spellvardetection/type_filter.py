# -*- coding: utf-8 -*-

import abc
import json

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
