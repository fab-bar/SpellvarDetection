import unittest

import spellvardetection.generator

class TestGeneratorFactory(unittest.TestCase):

    def test_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator("Unknown Type", {})


    def test_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator("lookup", {})

    def test_factory_for_lookup_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("lookup",
                                                                 {'dictionary': {'cat': set('abc')}}),
            spellvardetection.generator.LookupGenerator
        )
