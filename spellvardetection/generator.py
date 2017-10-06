# -*- coding: utf-8 -*-

import multiprocessing
import abc

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
        'lookup': (['dictionary'], lambda: LookupGenerator(options['dictionary']))
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
