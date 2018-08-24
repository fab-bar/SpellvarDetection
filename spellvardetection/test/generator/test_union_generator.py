import unittest

from spellvardetection.generator import _AbstractCandidateGenerator, GeneratorUnion

class MockGenerator(_AbstractCandidateGenerator):

    def __init__(self, candidates):
        self.candidates = candidates

    def getCandidatesForWord(self, word):
        return self.candidates

class TestGeneratorUnion(unittest.TestCase):

    def test_getCandidates(self):

        generator = GeneratorUnion([MockGenerator(['rat']), MockGenerator(['hat'])])
        self.assertEqual(generator.getCandidatesForWords(['cat', 'dog']), {'cat': set(['rat', 'hat']), 'dog': set(['rat', 'hat'])})
