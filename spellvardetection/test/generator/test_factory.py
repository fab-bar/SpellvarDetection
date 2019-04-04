import unittest

import spellvardetection.generator

class TestGeneratorFactory(unittest.TestCase):

    def test_factory_with_unknown_type(self):

        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator({"type": "Unknown Type"})


    def test_factory_with_missing_options(self):

        with self.assertRaises(ValueError):
            spellvardetection.generator.createCandidateGenerator({"type": "lookup"})

    def test_factory_for_union_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({"type": "union", "options": {"generators": []}}),
            spellvardetection.generator.GeneratorUnion
        )

    def test_factory_for_union_generator_with_generator_description_as_json(self):

        generator = spellvardetection.generator.createCandidateGenerator({"type": "union", "options": {"generators": [
            '{"type": "levenshtein"}',
            '{"type": "lookup", "options": {"spellvar_dictionary": {"cat": ["abc"]}}}'
        ]}})

        self.assertIsInstance(
            generator,
            spellvardetection.generator.GeneratorUnion
        )
        self.assertIsInstance(
            generator.generators[0],
            spellvardetection.generator.LevenshteinGenerator
        )
        self.assertIsInstance(
            generator.generators[1],
            spellvardetection.generator.LookupGenerator
        )

    def test_factory_for_lookup_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({
                "type": "lookup",
                "options": {'spellvar_dictionary': {'cat': set('abc')}}}),
            spellvardetection.generator.LookupGenerator
        )

    def test_factory_for_gent_gml_simplification_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({
                "type": "gent_gml_simplification",
                "options": {'dictionary': ['cat']}}),
            spellvardetection.generator.GentGMLSimplificationGenerator
        )

    def test_factory_for_simplification_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({
                "type": "simplification",
                "options": {'dictionary': ['cat'], 'ruleset': [('b', 'a'), ('g', 'gh')]}}),
            spellvardetection.generator.SimplificationGenerator
        )

    def test_factory_for_simplification_generator_with_generator_object(self):

        generator = spellvardetection.generator.createCandidateGenerator({
            "type": "simplification",
                "options": {
                    'dictionary': ['cat'],
                    'ruleset': [('b', 'a'), ('g', 'gh')],
                    'generator': spellvardetection.generator.LookupGenerator({'cat': set(['catt', 'caat'])})}})
        self.assertIsInstance(generator.generator, spellvardetection.generator._AbstractCandidateGenerator)

    def test_factory_for_simplification_generator_with_generator_description(self):

        generator = spellvardetection.generator.createCandidateGenerator({
            "type": "simplification",
                "options": {
                    'dictionary': ['cat'],
                    'ruleset': [('b', 'a'), ('g', 'gh')],
                    'generator': {'type': 'lookup',
                                  'options': {
                                      'spellvar_dictionary': {'cat': set(['catt', 'caat'])}
                                  }}}})
        self.assertIsInstance(generator.generator, spellvardetection.generator._AbstractCandidateGenerator)

    def test_factory_for_simplification_generator_with_generator_description_as_json(self):

        generator = spellvardetection.generator.createCandidateGenerator({
            "type": "simplification",
                "options": {
                    'dictionary': ['cat'],
                    'ruleset': [('b', 'a'), ('g', 'gh')],
                    'generator': """{"type": "lookup",
                                  "options": {
                                      "spellvar_dictionary": {"cat": ["catt", "caat"]}
                                  }}"""
                }})
        self.assertIsInstance(generator.generator, spellvardetection.generator._AbstractCandidateGenerator)

    def test_factory_for_levenshtein_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({
                "type": "levenshtein",
                "options": {'dictionary': ['cat']}}),
            spellvardetection.generator.LevenshteinGenerator
        )


    def test_factory_for_levenshteinnormalized_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({
                "type": "levenshtein_normalized",
                 "options": {'dictionary': ['cat']}}),
            spellvardetection.generator.LevenshteinNormalizedGenerator
        )

    def test_factory_for_proxinette_generator(self):

        self.assertIsInstance(
            spellvardetection.generator.createCandidateGenerator({
                "type": "proxinette",
                "options": {'dictionary': ['cat']}}),
            spellvardetection.generator.ProxinetteGenerator
        )
