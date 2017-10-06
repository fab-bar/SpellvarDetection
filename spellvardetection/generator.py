# -*- coding: utf-8 -*-

import multiprocessing
import abc
import re

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
def createCandidateGenerator(generator_type, options):
    """Factory for candidate generators."""

    generator_types = {
        'lookup': (
            ['dictionary'],
            lambda: LookupGenerator(options['dictionary'])
        ),
        'gent_gml_simplification': (
            ['dictionary'],
            lambda: GentGMLSimplificationGenerator(options['dictionary'], options.get('generator', None))
        )
    }

    if generator_type not in generator_types:
        raise ValueError('No candidate generator of type "' + generator_type + '" exists.')

    if not all(name in options for name in generator_types[generator_type][0]):
        raise ValueError('Missing options for generator of type ' + generator_type)

    return generator_types[generator_type][1]()

### Generators
class LookupGenerator(_AbstractCandidateGenerator):
    """A spelling variant generator based on a simple dictionary lookup"""

    def __init__(self, candidate_dictionary):

        self.candidate_dictionary = candidate_dictionary

    def getCandidatesForWord(self, word):

        return self.candidate_dictionary.get(word, set())


class _AbstractSimplificationGenerator(_AbstractCandidateGenerator):

    def __init__(self, dictionary, generator=None):

        self.simpl_candidates = {}
        for word in dictionary:
            simpl_word = self.__apply_rules(word)
            if simpl_word not in self.simpl_candidates:
               self.simpl_candidates[simpl_word] = set()
            self.simpl_candidates[simpl_word].add(word)

        self.generator = None
        if generator is not None:
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
