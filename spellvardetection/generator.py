# -*- coding: utf-8 -*-

import multiprocessing
import abc
import re
import collections
import math
import functools
from typing import Sequence

from spellvardetection.lib.lev_aut import DictAutomaton
import spellvardetection.lib.util
from spellvardetection.type_filter import _AbstractTypeFilter
from spellvardetection.util.feature_extractor import FeatureExtractorMixin, NGramExtractor

### The common interface for candidate generators
class _AbstractCandidateGenerator(metaclass=abc.ABCMeta):

    max_processes = 1

    def setMaxProcesses(self, processes):
        self.max_processes = processes

    def __getWordCandidatesPair(self, word):
        return (word, self.getCandidatesForWord(word))

    @abc.abstractmethod
    def getCandidatesForWord(self, word):  # pragma: no cover
        pass

    def setDictionary(self, dictionary: set):

        self.dictionary = dictionary

    def getCandidatesForWords(self, words):

        ## only use multiprocessing if number of max_processes is not 1
        if self.max_processes == 1:
            return {
                word: candidates for word, candidates in map(self.__getWordCandidatesPair, words)
            }
        else:
            with multiprocessing.Pool(self.max_processes) as pool:
                return {
                    word: candidates for word, candidates in pool.map(
                        functools.partial(spellvardetection.lib.util._unwrap_self, function_name="_AbstractCandidateGenerator__getWordCandidatesPair"),
                        zip([self]*len(words), words))
                }

class GeneratorUnion(_AbstractCandidateGenerator):

    name = 'union'

    def __init__(self,
                 generators: Sequence[_AbstractCandidateGenerator],
                 dictionary: set=None):

        self.generators = generators

        if dictionary is not None:
            self.setDictionary(dictionary)

    def getCandidatesForWord(self, word):

        return set().union(*[generator.getCandidatesForWord(word) for generator in self.generators])

    def setDictionary(self, dictionary: set):

        for generator in self.generators:
            generator.setDictionary(dictionary)

class GeneratorPipeline(_AbstractCandidateGenerator):

    name = 'pipeline'

    def __init__(self,
                 generator: _AbstractCandidateGenerator,
                 type_filter: _AbstractTypeFilter,
                 dictionary: set=None):

        self.generator = generator
        self.type_filter = type_filter

        if dictionary is not None:
            self.setDictionary(dictionary)

    def getCandidatesForWord(self, word):
        candidates = self.generator.getCandidatesForWord(word)
        return self.type_filter.filterCandidates(word, candidates)

    def setDictionary(self, dictionary: set):

        self.generator.setDictionary(dictionary)

### Generators
class LookupGenerator(_AbstractCandidateGenerator):
    """A spelling variant generator based on a simple dictionary lookup"""

    name = 'lookup'

    def __init__(self,
                 spellvar_dictionary: dict
    ):

        self.candidate_dictionary = {type_: frozenset(variants)
                                     for type_, variants in spellvar_dictionary.items()}

    def getCandidatesForWord(self, word):

        return self.candidate_dictionary.get(word, frozenset())


class _AbstractSimplificationGenerator(_AbstractCandidateGenerator):

    def __init__(self,
                 dictionary: set=None,
                 generator: _AbstractCandidateGenerator=None):

        self.generator = generator

        if dictionary is not None:
            self.setDictionary(dictionary)

    @abc.abstractmethod
    def __apply_rules(self, word):  # pragma: no cover
        pass


    def getCandidatesForWord(self, word):

        if not hasattr(self, 'simpl_candidates'):
            raise RuntimeError("Dictionary has to be set for generator of type " + self.name)

        simpl_words = [self.__apply_rules(word)]
        if self.generator is not None:
            simpl_words.extend(self.generator.getCandidatesForWord(simpl_words[0]))

        return set.union(*[self.simpl_candidates.get(simpl_word, set([])) for simpl_word in simpl_words]).difference([word])

    def setDictionary(self, dictionary: set):

        self.simpl_candidates = {}
        for word in dictionary:
            simpl_word = self.__apply_rules(word)
            if simpl_word not in self.simpl_candidates:
               self.simpl_candidates[simpl_word] = set()
            self.simpl_candidates[simpl_word].add(word)

        if self.generator is not None:
            self.generator.setDictionary(self.simpl_candidates.keys())

### A simplification generator with the rules from Koleva et al. 2017 (https://doi.org/10.1075/ijcl.22.1.05kol)
class GentGMLSimplificationGenerator(_AbstractSimplificationGenerator):

    name = 'gent_gml_simplification'

    def _AbstractSimplificationGenerator__apply_rules(self, word):

        ### this code has been provided by Melissa Farasyn
        rule1 = re.sub('c[k]?(?!h)', 'k', word)
        rule2 = re.sub('lyk', 'lik', rule1)
        rule3 = re.sub('lych', 'lich', rule2)
        rule4 = re.sub('lig', 'lyg', rule3)
        rule5 = re.sub('th(?!e[iye]?t|aft|alv|ert)', 't', rule4)
        rule6 = re.sub(r'\b.f[f]?te\b', 'efte', rule5)
        rule7 = re.sub('(?<![ng])g(?![ght])', 'gh', rule6)
        rule8 = re.sub('ggh(?!t)', 'gh', rule7)
        rule9 = re.sub('(?<!n)g[h]?t', 'cht', rule8)
        rule10 = re.sub('[aA][iye]', 'a', rule9)
        rule11 = re.sub('(?<!gh)ei', 'ey', rule10)
        rule12 = re.sub('(?<!gh)(?<!b)ee', 'ey', rule11)
        rule13 = re.sub('iy', 'i', rule12)
        rule14 = re.sub('(?<![xi])ij', 'i', rule13)
        rule15 = re.sub('o[ei]', 'oy', rule14)
        rule16 = re.sub(r'\beyne(?=\b)', 'ene', rule15)
        rule17 = re.sub(r'\beyne(?=[nrm]e\b)', 'ene', rule16)
        rule18 = re.sub(r'\beyne(?=[nrms]\b)', 'ene', rule17)
        rule19 = re.sub(r'(?<![AaEeIiUuOoYy])y(?![aeiuoyg])', 'i', rule18)
        rule20 = re.sub(r'(?<!\b)dt(?=\b)', 't', rule19)
        rule21 = re.sub(r'(?<!\b.n)(?<!\b)d(?![AaEeIiUuOo])(?=\b)', 't', rule20)
        rule22 = re.sub(r'ou[uv]', 'ouw', rule21)
        rule23 = re.sub('uul', 'vul', rule22)
        rule24 = re.sub(r'\bu[v]', 'vu', rule23)
        rule25 = re.sub(r'(?<=[AaEeIiUuOo])v(?=[AaEeIiUuOo])', 'u', rule24)
        last_rule = re.sub(r'(?<=\b).nd[e]?(?=\b)', 'vnde', rule25)

        return last_rule


class SimplificationGenerator(_AbstractSimplificationGenerator):

    name='simplification'

    def _AbstractSimplificationGenerator__apply_rules(self, word):

        for lhs, rhs in self.simplification_rules:
            word = word.replace(lhs, rhs)

        return word

    def __init__(self,
                 ruleset: list,
                 dictionary: set=None,
                 generator: _AbstractCandidateGenerator=None):

        simplification_rules = ruleset

        ## sort substitution rules internally: left side should be < then right side (except for deletion rules)
        simplification_rules = [sorted(rule, key=lambda t: (-len(t), t)) for rule in simplification_rules]
        ## process the rules: resolve competing rules, e.g. (i→j) and (i→y), should become (i→y), (j→y)
        rule_dict = collections.defaultdict(set)
        for lhs, rhs in simplification_rules:
            rule_dict[lhs].add(rhs)
        self.simplification_rules = []
        while rule_dict:
            lhs = sorted(rule_dict.keys())[0]
            rhs_list = rule_dict.pop(lhs)
            rhsides = sorted(rhs_list)
            target = rhsides.pop()
            self.simplification_rules.append((lhs, target))
            for rhs in rhsides:
                rule_dict[rhs].add(target)
        ## sort rules by length and alphabet (to apply deletion rules first)
        self.simplification_rules = sorted(self.simplification_rules, key=lambda t: (-len(t[0]), t[0]))

        super().__init__(dictionary, generator)

class _LevenshteinAutomatonGenerator(_AbstractCandidateGenerator):

    def __init__(self,
                 dictionary: set=None,
                 transposition=False, merge_split=False, repetitions=False):

        self.transposition = transposition
        self.merge_split = merge_split
        self.repetitions = repetitions

        if dictionary is not None:
            self.setDictionary(dictionary)

    def _getCandidatesForWord(self, word, distance):

        if not hasattr(self, 'dict_automaton'):
            raise RuntimeError("Dictionary has to be set for generator of type " + self.name)

        cands = self.dict_automaton.fuzzySearch(word, distance, transposition=self.transposition, merge_split=self.merge_split, repetitions=self.repetitions)

        if word in cands:
            cands.remove(word)

        return cands

    def setDictionary(self, dictionary: set):

        self.dict_automaton = DictAutomaton(dictionary)


class LevenshteinGenerator(_LevenshteinAutomatonGenerator):

    name = 'levenshtein'

    def __init__(self, dictionary: set=None,
                 max_dist=2,
                 transposition=False, merge_split=False, repetitions=False):

        super().__init__(dictionary, transposition, merge_split, repetitions)

        self.max_dist = max_dist

    def getCandidatesForWord(self, word):

        return super()._getCandidatesForWord(word, self.max_dist)


class LevenshteinNormalizedGenerator(_LevenshteinAutomatonGenerator):

    name = 'levenshtein_normalized'

    def __init__(self, dictionary: set=None,
                 dist_thresh=0.1, no_zero_dist=True,
                 transposition=False, merge_split=False, repetitions=False,
                 max_dist=5):

        super().__init__(dictionary, transposition, merge_split, repetitions)

        self.dist_thresh = dist_thresh
        self.no_zero_dist = no_zero_dist
        self.max_dist = max_dist

    def getCandidatesForWord(self, word):

        dist = math.floor(self.dist_thresh*len(word))
        if self.no_zero_dist:
            dist = max(1, dist)

        return super()._getCandidatesForWord(word, min(self.max_dist, dist))


class _SetsimilarityGenerator(_AbstractCandidateGenerator):

    def __init__(self,
                 featureset_extractor: FeatureExtractorMixin,
                 dictionary: set=None, sim_thresh=0.2, add_similarity=False):

        self.featureset_extractor = featureset_extractor
        self.sim_thresh = sim_thresh
        self.add_similarity = add_similarity

        if dictionary is not None:
            self.setDictionary(dictionary)

    @abc.abstractmethod
    def getSetsim(self, seta, setb):  # pragma: no cover
        pass

    def getCandidatesForWord(self, texttype):

        if not hasattr(self, 'dictionary'):
            raise RuntimeError("Dictionary has to be set for generator of type " + self.name)

        texttype_feat = self.featureset_extractor.extractFeaturesFromDatapoint(texttype)
        sim_cands = set()
        for feat in texttype_feat:
            if feat in self.feature_known:
                sim_cands.update([self.dictionary[index] for index in self.feature_known[feat]])

        sim_cands.discard(texttype)
        candidates = [(cand,
                       self.getSetsim(
                           texttype_feat,
                           self.featureset_extractor.extractFeaturesFromDatapoint(cand)))
        for cand in sim_cands]
        candidates = filter(lambda cand: cand[1] >= self.sim_thresh, candidates)

        if not self.add_similarity:
            candidates = map(lambda cand: cand[0], candidates)

        return set(candidates)

    def setDictionary(self, dictionary: set):

        self.dictionary = dictionary
        self.feature_known = {}

        for index, k_type in enumerate(self.dictionary):
            k_feat = self.featureset_extractor.extractFeaturesFromDatapoint(k_type)
            for feat in k_feat:
                if feat not in self.feature_known:
                    self.feature_known[feat] = []
                self.feature_known[feat].append(index)


class ProxinetteGenerator(_SetsimilarityGenerator):

    name = 'proxinette'

    def create(dictionary: set=None, sim_thresh=0.01,
               feature_extractor: FeatureExtractorMixin=None,
               add_similarity=False):

        if feature_extractor is None:
            feature_extractor = NGramExtractor(
                min_ngram_size=3,
                max_ngram_size=float('inf'),
                skip_size=0,
                gap='',
                bow='$',
                eow='$'
            )

        return ProxinetteGenerator(feature_extractor,
                                   dictionary, sim_thresh,
                                   add_similarity)


    def getSetsim(self, seta, setb):

        intersection = set.intersection(seta, setb)
        intersection_weightedsum = sum([1/float(len(self.feature_known[feat])) for feat in intersection])
        seta_cardinality = len(seta)
        return intersection_weightedsum/float(seta_cardinality)


class _AbstractJaccardSimilarityGenerator(_SetsimilarityGenerator):

    @abc.abstractmethod
    def _getWeightedSum(self, feature_set):  # pragma: no cover
        pass

    @classmethod
    def create(cls, dictionary: set=None, sim_thresh=0.2,
               feature_extractor: FeatureExtractorMixin=None,
               add_similarity=False):

        if feature_extractor is None:
            feature_extractor = NGramExtractor(
                min_ngram_size=2,
                max_ngram_size=2,
                skip_size=1,
                gap='|',
                bow='$',
                eow='$',
                pad_ngrams=False
            )

        return cls(
            feature_extractor,
            dictionary, sim_thresh,
            add_similarity)


    def setDictionary(self, dictionary: set):

        super().setDictionary(dictionary)
        self.feat_weights = self._getFeatureWeights()

    def getSetsim(self, seta, setb):

        intersection_wsum = self._getWeightedSum(set.intersection(seta, setb))
        union_wsum = self._getWeightedSum(set.union(seta, setb))
        return intersection_wsum/float(union_wsum)

    def _getWeightedSum(self, feature_set):

        return sum([self.feat_weights.get(feat, self.default_weight) for feat in feature_set])

class JaccardSimilarityGenerator(_AbstractJaccardSimilarityGenerator):

    name = 'jaccard'

    default_weight = 1

    def _getFeatureWeights(self):

        ## everything has (the default) weight 1
        return {}

    def _getWeightedSum(self, feature_set):

        # features are not weighted - sum is just the length of the feature set
        # faster than the generic version of _getWeightedSum
        return len(feature_set)

class FrequencyWeightedJaccardSimilarityGenerator(_AbstractJaccardSimilarityGenerator):

    name = 'frequency_wjaccard'

    default_weight = 1

    def _getFeatureWeights(self):

        # each feature is weighted by its relative frequency in the dictionary
        return {feat: (1 - len(self.feature_known.get(feat))/float(len(self.dictionary))) for feat in self.feature_known.keys()}
