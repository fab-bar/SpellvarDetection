import unittest

import spellvardetection.generator

class TestGeneratorFactory(unittest.TestCase):

    def test_factory_with_unknown_type(self):


        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator("Unknown Type", {})
