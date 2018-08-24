import unittest

from spellvardetection.lib.undir_spsim import UndirSpSim

class TestUndirSpSim(unittest.TestCase):

    def setUp(self):

        self.sim = UndirSpSim(bow="|", eow="|")

    def _test_similarity(self, sim_phase, sim_phenomenal):

        self.assertEquals(
            self.sim('phase', 'fase'),
            sim_phase
        )
        self.assertEquals(
            self.sim('phase', 'fase'),
            self.sim('fase', 'phase')
        )

        self.assertEquals(
            self.sim('phenomenal', 'fenomenal'),
            sim_phenomenal
        )
        self.assertEquals(
            self.sim('phenomenal', 'fenomenal'),
            self.sim('fenomenal', 'phenomenal')
        )

    def test_similarity_without_training(self):

        self._test_similarity(0.6, 0.8)


    def test_similarity_with_specific_rightcontext(self):

        self.sim.learn([('phase', 'fase')])

        self._test_similarity(1, 0.8)

    def test_similarity_with_unspecific_rightcontext(self):

        self.sim.learn([('phase', 'fase')])
        self.sim.learn([('photo', 'foto')])

        self._test_similarity(1, 1)

    def test_similarity_with_caret(self):

        ## ^ is used to encode beginning of word in spsim

        self.sim.learn([('phase', 'fase')])

        self.assertEquals(
            self.sim('^phas', '^fas'),
            0.6
        )

    def test_similarity_with_dollar(self):

        ## $ is used to encode end of word in spsim

        self.sim.learn([('neulic', 'neulig')])

        self.assertEquals(
            self.sim('ic$a', 'ig$a'),
            0.75
        )
