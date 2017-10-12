# -*- coding: utf-8 -*-

import multiprocessing
import abc
import re
import collections
import math

from spellvardetection.lib.lev_aut import DictAutomaton
import spellvardetection.lib.util

### this function is used to allow to use an instance method in pool.map
### based on http://www.rueckstiess.net/research/snippets/show/ca1d7d90
def _unwrap_self(arg, **kwarg):
    return type(arg[0])._AbstractCandidateGenerator__getWordCandidatesPair(*arg, **kwarg)

### The common interface for candidate generators
class _AbstractCandidateGenerator(metaclass=abc.ABCMeta):

    def __getWordCandidatesPair(self, word):
        return (word, self.getCandidatesForWord(word))

    @abc.abstractmethod
    def getCandidatesForWord(self, word):
        pass

    def getCandidatesForWords(self, words):

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            return {word: candidates for word, candidates in pool.map(_unwrap_self, zip([self]*len(words), words))}

## Factory for generators
createCandidateGenerator = spellvardetection.lib.util.createFactory("generator",
    {
        'lookup': (
            ['dictionary'],
            lambda options: LookupGenerator(options['dictionary'])
        ),
        'simplification': (
            ['dictionary', 'ruleset'],
            lambda options: SimplificationGenerator(options['ruleset'], options['dictionary'], options.get('generator', None))
        ),
        'gent_gml_simplification': (
            ['dictionary'],
            lambda options: GentGMLSimplificationGenerator(options['dictionary'], options.get('generator', None))
        ),
        'levenshtein': (
            ['dictionary'],
            lambda options: LevenshteinGenerator(options['dictionary'], options.get('max_dist', 2), options.get('transposition', False), options.get('merge_split', False), options.get('repetitions', False))
        ),
        'levenshtein_normalized': (
            ['dictionary'],
            lambda options: LevenshteinNormalizedGenerator(options['dictionary'], options.get('dist_thresh', 0.1), options.get('no_zero_dist', True), options.get('transposition', False), options.get('merge_split', False), options.get('repetitions', False))
        )
    }
)

### Generators
class LookupGenerator(_AbstractCandidateGenerator):
    """A spelling variant generator based on a simple dictionary lookup"""

    def __init__(self, candidate_dictionary):
        self.candidate_dictionary = spellvardetection.lib.util.load_from_file_if_string(candidate_dictionary)

    def getCandidatesForWord(self, word):

        return self.candidate_dictionary.get(word, set())


class _AbstractSimplificationGenerator(_AbstractCandidateGenerator):

    def __init__(self, dictionary, generator=None):

        dictionary = spellvardetection.lib.util.load_from_file_if_string(dictionary)

        self.simpl_candidates = {}
        for word in dictionary:
            simpl_word = self.__apply_rules(word)
            if simpl_word not in self.simpl_candidates:
               self.simpl_candidates[simpl_word] = set()
            self.simpl_candidates[simpl_word].add(word)

        self.generator = None
        if generator is not None:
            generator = spellvardetection.lib.util.load_from_file_if_string(generator)
            self.generator = createCandidateGenerator(generator[0], {**generator[1], **{'dictionary': self.simpl_candidates.keys()}})

    @abc.abstractmethod
    def __apply_rules(self, word):
        pass


    def getCandidatesForWord(self, word):

        simpl_words = [self.__apply_rules(word)]
        if self.generator is not None:
            simpl_words.extend(self.generator.getCandidatesForWord(simpl_words[0]))

        return set.union(*[self.simpl_candidates[simpl_word] for simpl_word in simpl_words]).difference([word])


### A simplification generator with the rules from Koleva et al. 2017 (https://doi.org/10.1075/ijcl.22.1.05kol)
class GentGMLSimplificationGenerator(_AbstractSimplificationGenerator):

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

    def _AbstractSimplificationGenerator__apply_rules(self, word):

        for lhs, rhs in self.simplification_rules:
            word = word.replace(lhs, rhs)

        return word

    def __init__(self, simplification_rules, dictionary, generator=None):

        simplification_rules = spellvardetection.lib.util.load_from_file_if_string(simplification_rules)

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

    def __init__(self, dictionary, transposition=False, merge_split=False, repetitions=False):

        self.dict_automaton = DictAutomaton(spellvardetection.lib.util.load_from_file_if_string(dictionary))
        self.transposition = transposition
        self.merge_split = merge_split
        self.repetitions = repetitions

    def _getCandidatesForWord(self, word, distance):

        cands = self.dict_automaton.fuzzySearch(word, distance, transposition=self.transposition, merge_split=self.merge_split, repetitions=self.repetitions)

        if word in cands:
            cands.remove(word)

        return cands

class LevenshteinGenerator(_LevenshteinAutomatonGenerator):

    def __init__(self, dictionary, max_dist, transposition=False, merge_split=False, repetitions=False):

        super().__init__(dictionary, transposition, merge_split, repetitions)

        self.max_dist = max_dist

    def getCandidatesForWord(self, word):

        return super()._getCandidatesForWord(word, self.max_dist)


class LevenshteinNormalizedGenerator(_LevenshteinAutomatonGenerator):

    def __init__(self, dictionary, dist_thresh, no_zero_dist=True, transposition=False, merge_split=False, repetitions=False):

        super().__init__(dictionary, transposition, merge_split, repetitions)

        self.dist_thresh = dist_thresh
        self.no_zero_dist = no_zero_dist

    def getCandidatesForWord(self, word):

        dist = math.floor(self.dist_thresh*len(word))
        if self.no_zero_dist:
            dist = max(1, dist)

        return super()._getCandidatesForWord(word, dist)
