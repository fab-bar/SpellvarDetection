import json

import unittest

import click
from click.testing import CliRunner

from sklearn.externals import joblib
from sklearn.dummy import DummyClassifier

import spellvardetection.cli
from spellvardetection.type_filter import createTypeFilter

class TestCLI(unittest.TestCase):

    def test_generate_without_lexicon(self):

        runner = CliRunner()
        result = runner.invoke(spellvardetection.cli.main, ['generate', '["vnd"]', '{"type": "levenshtein", "options": {"max_dist": 1}}'])
        self.assertEquals('Dictionary has to be set for generator of type levenshtein\n', result.output)

    def test_generate_candidates(self):

        runner = CliRunner()
        result = runner.invoke(spellvardetection.cli.main, ['generate', '["vnd"]', '{"type": "levenshtein", "options": {"max_dist": 1}}', '-d', '["und", "unde", "vnde", "vns"]'])

        result_dict = json.loads(result.output)
        result_dict["vnd"] = set(result_dict["vnd"])
        self.assertEquals(result_dict, {"vnd": set(["und", "vnde", "vns"])})

    def test_generate_candidates_with_internal_generator_from_file(self):

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('generator.json', 'w') as f:
                f.write('{"type": "levenshtein", "options": {"max_dist": 1}}')

            result = runner.invoke(spellvardetection.cli.main, ['generate', '["vnd"]', """{"type": "simplification",
                "options": {
                        "dictionary": ["unde", "und", "vns", "vnde"],
                        "ruleset": [["u", "v"]],
                        "generator": "generator.json"
                 }
            }"""])

            result_dict = json.loads(result.output)
            result_dict["vnd"] = set(result_dict["vnd"])
            self.assertEquals(result_dict, {"vnd": set(["unde", "und", "vnde", "vns"])})


    def test_generate_candidates_with_union_of_generators_from_file(self):

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('leven_generator.json', 'w') as f:
                f.write('{"type": "levenshtein", "options": {"max_dist": 1}}')
            with open('lookup_generator.json', 'w') as f:
                f.write('{"type": "lookup", "options": {"spellvar_dictionary": {"vnd": ["unde"]}}}')

            result = runner.invoke(spellvardetection.cli.main, ['generate', '["vnd"]', """{"type": "union",
                "options": {
                        "dictionary": ["unde", "und", "vns", "vnde"],
                        "generators": ["leven_generator.json", "lookup_generator.json"]
                 }
            }"""])

            result_dict = json.loads(result.output)
            result_dict["vnd"] = set(result_dict["vnd"])
            self.assertEquals(result_dict, {"vnd": set(["unde", "und", "vnde", "vns"])})

    def test_filter_candidates(self):

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('brown.txt', 'w') as f:
                f.write('1010	und	14\n1010	vnd	16\n1010	vnde	1604\n1010	unde	897\n11110	vns	31\n')

            result = runner.invoke(spellvardetection.cli.main, ['filter', '{"vnd": ["und", "vns"]}', '{"type": "cluster", "options": {"cluster_type": "brown", "cluster_file": "brown.txt"}}'])

            result_dict = json.loads(result.output)
            result_dict["vnd"] = set(result_dict["vnd"])
            self.assertEquals(result_dict, {"vnd": set(["und"])})

    def test_train_filter(self):

        runner = CliRunner()
        with runner.isolated_filesystem():

            result = runner.invoke(spellvardetection.cli.main, [
                'train', 'filter',
                '{"type": "sklearn", "options": {"classifier_clsname": "sklearn.dummy.DummyClassifier", "classifier_params": {"strategy": "constant", "constant": 0}, "feature_extractors": [{"type": "surface", "name": "surface"}]}}',
                'dummy.model',
                '[["under", "vnder"]]',
                '[["hans", "hand"]]'])

            clf = joblib.load('dummy.model')
            self.assertTrue(isinstance(clf.classifier, DummyClassifier))


    def test_train_filter_with_context_extractor(self):

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('embeddings.txt', 'w') as f:
                f.write('under 1 0 0\nvnder 1 0 0\nhans 0 1 0\nhand 0 0 1\n')

            result = runner.invoke(spellvardetection.cli.main, [
                'train', 'filter',
                '{"type": "sklearn", "options": {"classifier_clsname": "sklearn.dummy.DummyClassifier", "classifier_params": {"strategy": "constant", "constant": 0}, "feature_extractors": [{"type": "context", "name": "context", "options": {"vector_type": "hyperwords", "vectorfile_name": "embeddings.txt"}}]}}',
                'dummy.model',
                '[["under", "vnder"]]',
                '[["hans", "hand"]]'])

            clf = joblib.load('dummy.model')
            self.assertTrue(isinstance(clf.classifier, DummyClassifier))


    def _train_filter(self, runner, feature_extractor_options, global_cache=None, positive_pairs=None, negative_pairs=None):

        if positive_pairs is None:
            positive_pairs = '[["hans", "hand"]]'
        if negative_pairs is None:
            negative_pairs = '[["under", "vnder"]]'

        cli_options = [
            'train', 'filter'
        ]
        if global_cache is not None:
            cli_options.extend(['-c', global_cache])
        cli_options.extend([
            '{"type": "sklearn", "options": {"classifier_clsname": "__svm__", "feature_extractors": [{"type": "surface", "name": "surface", ' + feature_extractor_options + '}]}}',
            'dummy.model',
            positive_pairs,
            negative_pairs,
        ])

        runner.invoke(spellvardetection.cli.main, cli_options)


    def _evaluate_trained_filter(self, type_, modelfile):

        ## test that the model will predict "vnd" "vns" as negative and "und", "vnd" as positive
        filter_ = createTypeFilter({"type": type_, "options": {"modelfile_name": modelfile}})
        result = filter_.filterCandidates("vnd", ["und", "vns"])
        self.assertEquals(result, set(["und"]))

    def test_train_filter_with_feature_cache(self):

        runner = CliRunner()
        with runner.isolated_filesystem():

            self._train_filter(runner,
                               ## features for "under, vnder" and "hans", "hand" are exchanged in the cache  - will train the opposite
                               '"cache": {"[\\"under\\", \\"vnder\\"]": [["nn", "ds"], ["nn", "ds", "$$"], ["aa", "nn", "ds"], ["ds", "$$"], ["ds"]], "[\\"hand\\", \\"hans\\"]": [["$$", "uv"], ["$$", "uv", "nn"], ["uv", "nn", "dd"], ["uv", "nn"], ["uv"]]}'
            )
            self._evaluate_trained_filter("sklearn", "dummy.model")


    def test_train_filter_with_feature_cache_from_file(self):

        runner = CliRunner()
        with runner.isolated_filesystem():

            with open('feature_cache.txt', 'w') as f:
                ## features for "under, vnder" and "hans", "hand" are exchanged in the cache  - will train the opposite
                f.write('{"[\\"under\\", \\"vnder\\"]": [["nn", "ds"], ["nn", "ds", "$$"], ["aa", "nn", "ds"], ["ds", "$$"], ["ds"]], "[\\"hand\\", \\"hans\\"]": [["$$", "uv"], ["$$", "uv", "nn"], ["uv", "nn", "dd"], ["uv", "nn"], ["uv"]]}')

            self._train_filter(runner,
                               '"cache": "feature_cache.txt"'
            )
            self._evaluate_trained_filter("sklearn", "dummy.model")


    def test_train_filter_with_global_feature_cache(self):

        runner = CliRunner()
        with runner.isolated_filesystem():

            self._train_filter(runner,
                               '"key": "ngrams"',
                               ## features for "under, vnder" and "hans", "hand" are exchanged - will train the opposite
                               '{"[\\"under\\", \\"vnder\\"]": {"ngrams": [["nn", "ds"], ["nn", "ds", "$$"], ["aa", "nn", "ds"], ["ds", "$$"], ["ds"]]}, "[\\"hand\\", \\"hans\\"]": {"ngrams": [["$$", "uv"], ["$$", "uv", "nn"], ["uv", "nn", "dd"], ["uv", "nn"], ["uv"]]}}'
            )
            self._evaluate_trained_filter("sklearn", "dummy.model")

    def test_train_filter_with_global_feature_cache_from_file(self):

        runner = CliRunner()
        with runner.isolated_filesystem():

            with open('feature_cache.txt', 'w') as f:
                ## features for "under, vnder" and "hans", "hand" are exchanged - will train the opposite
                f.write('{"[\\"under\\", \\"vnder\\"]": {"ngrams": [["nn", "ds"], ["nn", "ds", "$$"], ["aa", "nn", "ds"], ["ds", "$$"], ["ds"]]}, "[\\"hand\\", \\"hans\\"]": {"ngrams": [["$$", "uv"], ["$$", "uv", "nn"], ["uv", "nn", "dd"], ["uv", "nn"], ["uv"]]}}')

            self._train_filter(runner,
                               '"key": "ngrams"',
                               'feature_cache.txt'
            )
            self._evaluate_trained_filter("sklearn", "dummy.model")

    def test_pipeline(self):

        runner = CliRunner()
        with runner.isolated_filesystem():

            self._train_filter(runner, '"key": "ngrams"',
                               positive_pairs='[["under", "vnder"]]',
                               negative_pairs='[["hans", "hand"]]'
            )

            result = runner.invoke(spellvardetection.cli.main, ['generate', '["vnd"]', """
            {"type": "pipeline", "options": {
               "generator": {"type": "levenshtein", "options": {"max_dist": 1}},
               "type_filter": {"type": "sklearn", "options": {"modelfile_name": "dummy.model"}}
            }}""", '-d', '["und", "unde", "vnde", "vns"]'])

            result_dict = json.loads(result.output)
            result_dict["vnd"] = set(result_dict["vnd"])
            self.assertEquals(result_dict, {"vnd": set(["und", "vnde"])})
