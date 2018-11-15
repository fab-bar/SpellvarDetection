import json

import unittest

import click
from click.testing import CliRunner

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
