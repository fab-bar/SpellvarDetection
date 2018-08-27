import unittest

import spellvardetection.generator

class TestGeneratorFactory(unittest.TestCase):

    def test_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator("Unknown Type", {})


    def test_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator("lookup", {})

    def test_factory_for_union_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator('union', {"generators": []}),
            spellvardetection.generator.GeneratorUnion
        )

    def test_factory_for_lookup_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("lookup",
                                                                 {'spellvar_dictionary': {'cat': set('abc')}}),
            spellvardetection.generator.LookupGenerator
        )

    def test_factory_for_gent_gml_simplification_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("gent_gml_simplification",
                                                                 {'dictionary': ['cat']}),
            spellvardetection.generator.GentGMLSimplificationGenerator
        )

    def test_factory_for_simplification_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("simplification",
                                                                 {'dictionary': ['cat'], 'ruleset': [('b', 'a'), ('g', 'gh')]}),
            spellvardetection.generator.SimplificationGenerator
        )

    def test_factory_for_levenshtein_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("levenshtein",
                                                                 {'dictionary': ['cat']}),
            spellvardetection.generator.LevenshteinGenerator
        )


    def test_factory_for_levenshteinnormalized_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("levenshtein_normalized",
                                                                 {'dictionary': ['cat']}),
            spellvardetection.generator.LevenshteinNormalizedGenerator
        )

    def test_factory_for_proxinette_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator("proxinette",
                                                                 {'dictionary': ['cat']}),
            spellvardetection.generator.ProxinetteGenerator
        )
