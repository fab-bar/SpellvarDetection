# -*- coding: utf-8 -*-

import abc
import importlib

from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.externals import joblib
from sklearn.svm import SVC

from imblearn.ensemble import BalancedBaggingClassifier

from spellvardetection.util.feature_extractor import createFeatureExtractor, SurfaceExtractor
import spellvardetection.lib.util


class _AbstractTypeFilter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def isPair(self, word, candidate):
        pass

    def filterCandidates(self, word, candidates):

        return set([candidate for candidate in candidates if self.isPair(word, candidate)])

class _AbstractTrainableTypeFilter(_AbstractTypeFilter):


    @abc.abstractmethod
    def train(self, positive_pairs, negative_pairs):
        pass

    @abc.abstractmethod
    def load(modelfile_name):
        pass

    @abc.abstractmethod
    def save(self, modelfile_name):
        pass


class SKLearnClassifierBasedTypeFilter(_AbstractTrainableTypeFilter, _BaseComposition, ClassifierMixin):

    name = 'sklearn'

    def create(modelfile_name):
        return SKLearnClassifierBasedTypeFilter.load(modelfile_name)


    def create_for_training(classifier_clsname, feature_extractors, classifier_params=None):

        ## instantiate classifier
        if classifier_clsname == '__svm__':
            classifier = SVC(gamma=0.1, C=2)
        elif classifier_clsname == '__bagging_svm__':
            classifier = BalancedBaggingClassifier(
                base_estimator=SVC(gamma=0.1, C=2),
                n_estimators=10,
                bootstrap=False,
                ratio='majority'
            )
        elif '.' in classifier_clsname:
            module_name, cls_name = classifier_clsname.rsplit('.', 1)
            module = importlib.import_module(module_name)
            classifier = getattr(module, cls_name)()
        else:
            classifier = globals()[classifier_clsname]()

        if classifier_params is not None:
            classifier.set_params(**classifier_params)

        ## instantiate feature extractors
        extractors = []
        for extractor in feature_extractors:
            extractor_object = createFeatureExtractor(extractor['type'], extractor.get('options', {}))
            if 'cache' in extractor:
                extractor_object.setFeatureCache(extractor['cache'], key=extractor.get('key', None))
            extractors.append((extractor['name'], extractor_object))

        return SKLearnClassifierBasedTypeFilter(classifier, extractors)


    def __init__(self, classifier=None, feature_extractors=None):

        if classifier is None:
            self.classifier = SVC()
        else:
            self.classifier = classifier

        if feature_extractors is None:
            self.feature_extractors = [('surface', SurfaceExtractor())]
        else:
            self.feature_extractors = feature_extractors


    def fit(self, X_data, Y_data=None):
        self._clf = Pipeline([
            ('features', FeatureUnion(transformer_list=self.feature_extractors)),
            ('clf', self.classifier),
        ])
        self._clf.fit(X_data, Y_data)

        return self

    def predict(self, X_data):
        try:
            getattr(self, "_clf")
        except AttributeError:
            raise RuntimeError("Classifier has to be trained!")

        return(self._clf.predict(X_data))

    def get_params(self, deep=True):

        features = self._get_params('feature_extractors', deep=deep)
        if deep:
            features = {**features, **{'classifier__' + key: value for key, value in self.classifier.get_params(deep=deep).items()}}

        return features

    def set_params(self, **params):

        self._set_params('feature_extractors', **params)


    def train(self, positive_pairs, negative_pairs):

        X = (positive_pairs + negative_pairs)
        Y = [1]*len(positive_pairs) + [0]*len(negative_pairs)

        self.fit(X, Y)

    def isPair(self, word, candidate):

        if self.predict([(word, candidate)]) == 1:
            return True
        else:
            return False

    def load(modelfile_name):
        return joblib.load(modelfile_name)

    def save(self, modelfile_name):
        joblib.dump(self, modelfile_name)


## Factory for filters
createTrainableTypeFilter = spellvardetection.lib.util.create_factory("trainable_type_filter", _AbstractTrainableTypeFilter, create_func='create_for_training')
createTypeFilter = spellvardetection.lib.util.create_factory("type_filter", _AbstractTypeFilter, create_func='create')
