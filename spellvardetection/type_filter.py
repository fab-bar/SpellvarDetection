# -*- coding: utf-8 -*-

import abc
import importlib
import math
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import typing

import joblib
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.svm import SVC

from imblearn.ensemble import BalancedBaggingClassifier

from spellvardetection.util.feature_extractor import FeatureExtractorMixin, SurfaceExtractor
import spellvardetection.lib.clusters
from spellvardetection.lib.undir_spsim import UndirSpSim
import spellvardetection.lib.util


class _AbstractTypeFilter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def isPair(self, word, candidate):  # pragma: no cover
        pass

    def filterCandidates(self, word, candidates):

        return set([candidate for candidate in candidates if self.isPair(word, candidate)])

class _AbstractTrainableTypeFilter(_AbstractTypeFilter):


    @abc.abstractmethod
    def train(self, positive_pairs, negative_pairs):  # pragma: no cover
        pass

    @abc.abstractmethod
    def load(modelfile_name):  # pragma: no cover
        pass

    @abc.abstractmethod
    def save(self, modelfile_name):  # pragma: no cover
        pass


class SKLearnClassifierBasedTypeFilter(_AbstractTrainableTypeFilter, _BaseComposition, ClassifierMixin):

    name = 'sklearn'

    def create(modelfile_name: os.PathLike):
        return SKLearnClassifierBasedTypeFilter.load(modelfile_name)


    def create_for_training(classifier_clsname,
                            feature_extractors: typing.Sequence[FeatureExtractorMixin],
                            classifier_params=None):

        ## instantiate classifier
        if classifier_clsname == '__svm__':
            classifier = SVC(gamma=0.1, C=2)
        elif classifier_clsname == '__bagging_svm__':
            classifier = BalancedBaggingClassifier(
                base_estimator=SVC(gamma=0.1, C=2),
                n_estimators=10,
                bootstrap=False,
                sampling_strategy='majority'
            )
        elif '.' in classifier_clsname:
            module_name, cls_name = classifier_clsname.rsplit('.', 1)
            module = importlib.import_module(module_name)
            classifier = getattr(module, cls_name)()
        else:
            classifier = globals()[classifier_clsname]()

        if classifier_params is not None:
            classifier.set_params(**classifier_params)

        extractors = [(str(idx), extractor) for idx, extractor in enumerate(feature_extractors)]

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


class ClusterTypeFilter(_AbstractTypeFilter):

    name = 'cluster'

    def create(cluster_type, cluster_file: os.PathLike, unknown_type=None, remove_candidates_without_cluster=False):

        clusters = spellvardetection.lib.clusters.WordClusters(cluster_type, cluster_file, unknown_type)
        return ClusterTypeFilter(clusters, remove_candidates_without_cluster)

    def __init__(self, clusters, remove_candidates_without_cluster=False):
        self.clusters = clusters
        self.remove_candidates_without_cluster = remove_candidates_without_cluster

    def isPair(self, word, candidate):

        if not self.clusters.hasCluster(word):
            return True
        elif not self.clusters.hasCluster(candidate):
            if self.remove_candidates_without_cluster:
                return False
            else:
                return True
        else:
            return self.clusters.inSameCluster(word, candidate)


class _SimilarityFilter(_AbstractTypeFilter):

    def __init__(self, similarity, sim_thresh=0.9):

        self.similarity = similarity
        self.sim_thresh = sim_thresh

    def isPair(self, word, candidate):

        return self.similarity(word, candidate) >= self.sim_thresh

class UndirSpSimTypeFilter(_SimilarityFilter, _AbstractTrainableTypeFilter):

    name = 'uspsim'

    def create(spsim_filename: os.PathLike,
               sim_thresh=0.9,
               ignore_case=True,
               ignore_accents=True,
               group_vowels=False,
               no_empty=False,
               bow='^',
               eow="$"):

        filter_ = UndirSpSimTypeFilter(None, sim_thresh)
        filter_.load(spsim_filename)
        return filter_


    def create_for_training(sim_thresh=0.9,
                            ignore_case=True,
                            ignore_accents=True,
                            group_vowels=False,
                            no_empty=False,
                            bow='^',
                            eow="$"):

        uspsim = UndirSpSim(ignore_case=ignore_case,
                            ignore_accents=ignore_accents,
                            group_vowels=group_vowels,
                            no_empty=no_empty,
                            bow=bow,
                            eow=eow)
        return UndirSpSimTypeFilter(uspsim, sim_thresh)

    def train(self, positive_pairs, negative_pairs):

        ### negative pairs are ignored (SpSim only uses positive pairs)
        self.similarity.learn(positive_pairs)

    def load(self, file_name):

        with open(file_name, 'rb') as infile:
            self.similarity = pickle.load(infile)

    def save(self, file_name):

        with open(file_name, 'wb') as outfile:
            pickle.dump(self.similarity, outfile)


class EditProbabilitiesTypeFilter(_SimilarityFilter):

    name = 'edit_probabilites'

    def similarity(self, word, candidate):

        alignment = filter(lambda e: e != 'IDD',
                                spellvardetection.lib.util.get_alignment(
                                    word, candidate,
                                    directed=False, conflate_id_pairs=True,
                                    empty_char=''
                                ))

        sim = 0
        for e in alignment:
            if e in self.probabilities:
                sim += self.probabilities[e]
            else:
                sim += self.default_probability

        return sim

    def __init__(self, probabilities: list, sim_thresh=0.9, default_probability=1.0):

        self.default_probability = math.log(default_probability)

        # convert chars into keys
        probabilities = [{'chars': ''.join(sorted([edit_op['char1'], edit_op['char2']])), 'probability': edit_op['probability']}
                         for edit_op in probabilities]

        if not all(map(lambda p: p['probability'] <= 1 and p['probability'] > 0, probabilities)):
            raise ValueError('Probabilities below or equal to 0 or above 1 are not allowed.')

        # double pairs
        if not len(set(map(lambda p: p['chars'], probabilities))) == len(probabilities):
            raise ValueError('Duplicate weights')

        self.probabilities = {
            edit_op['chars']: math.log(edit_op['probability']) for edit_op in probabilities
        }

        self.sim_thresh = math.log(sim_thresh)
