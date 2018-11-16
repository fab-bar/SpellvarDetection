import json

import unittest

import click
from click.testing import CliRunner

from sklearn.externals import joblib
from sklearn.dummy import DummyClassifier

import spellvardetection.cli

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
