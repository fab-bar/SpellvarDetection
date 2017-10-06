import unittest

from spellvardetection.generator import LookupGenerator

class TestLookupGenerator(unittest.TestCase):


    def setUp(self):
        self.generator = LookupGenerator(
            {
                'cat': set(['catt', 'caat'])
            }
        )

    def test_get_candidates_for_known_type(self):

        self.assertEquals(self.generator.getCandidatesForWord('cat'),
                          set(['catt', 'caat']))

    def test_get_candidates_for_unknown_type(self):

        self.assertEquals(self.generator.getCandidatesForWord('unknown type'),
                          set([]))
