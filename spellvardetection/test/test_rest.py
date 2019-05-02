import json
import os
import tempfile
import unittest

from spellvardetection.rest import create_app

class TestRest(unittest.TestCase):

    def setUp(self):

        # create temporary folder for db
        self.db_dir = tempfile.TemporaryDirectory()
        # set settings for app with app factory
        app = create_app(
            {
                'TESTING': True,
                'DATABASE': self.db_dir.name
            }
        )

        self.test_client = app.test_client()
        self.test_runner = app.test_cli_runner()
        ## add test dictionary and generator
        self.test_runner.invoke(args=["db", "add-dictionary", "gml", '["und", "vnde", "vnnde", "unde", "vns"]'])
        self.test_runner.invoke(args=["db", "add-generator", "lev1", '{"type": "levenshtein", "options": {"max_dist": 1, "repetitions": "True"}}'])

    def test_generate_candidates(self):

        response = self.test_client.get("/generate/lev1/gml/vnd")
        self.assertEquals(set(json.loads(response.data)),
                          set(["vns", "und", "vnnde", "vnde"]))

    def test_generate_candidates_for_multiple(self):

        response = self.test_client.post("/generate/lev1/gml", json=["vnd", "vnnd"])
        self.assertEquals(
            {
                "vnd": set(["und", "vnde", "vnnde", "vns"]),
                "vnnd": set(["und", "vnde", "vnnde", "vns"])

            },
            {key: set(value) for key, value in json.loads(response.data).items()}
        )
