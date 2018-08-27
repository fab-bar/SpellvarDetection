import unittest

import spellvardetection.test.MockClasses as MockClasses

from spellvardetection.generator import GeneratorPipeline

class TestGeneratorPipeline(unittest.TestCase):

    def test_getCandidates(self):

        generator = GeneratorPipeline(MockClasses.Generator(['rat', 'hat', 'flat']), MockClasses.TypeFilter(['flat']))
        self.assertEqual(generator.getCandidatesForWords(['cat', 'dog']), {'cat': set(['rat', 'hat']), 'dog': set(['rat', 'hat'])})
