import unittest

import spellvardetection.test.MockClasses as MockClasses

from spellvardetection.generator import GeneratorUnion

class TestGeneratorUnion(unittest.TestCase):

    def test_getCandidates(self):

        generator = GeneratorUnion([MockClasses.Generator(['rat']), MockClasses.Generator(['hat'])])
        self.assertEqual(generator.getCandidatesForWords(['cat', 'dog']), {'cat': set(['rat', 'hat']), 'dog': set(['rat', 'hat'])})
