from spellvardetection.generator import _AbstractCandidateGenerator
from spellvardetection.type_filter import _AbstractTypeFilter

class Generator(_AbstractCandidateGenerator):

    def __init__(self, candidates):
        self.candidates = candidates

    def getCandidatesForWord(self, word):
        return self.candidates

class TypeFilter(_AbstractTypeFilter):

    def __init__(self, filter_types):
        self.filter_types = set(filter_types)

    def isPair(self, word, candidate):
        return candidate not in self.filter_types


class Cluster:

    def __init__(self, vocabulary, pairs):

        self.voc = vocabulary
        self.pairs = pairs

    def hasCluster(self, word):

        if word in self.voc:
            return True

    def inSameCluster(self, word, candidate):

        return (word, candidate) in self.pairs
