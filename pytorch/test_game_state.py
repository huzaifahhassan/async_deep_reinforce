# Identical check to ../game_state_test.py, run against this folder's
# GameState -- it's the same class, just with a different ROM path
# resolution, so the same behavior should hold.
import unittest
import numpy as np

from game_state import GameState


class TestSequenceFunctions(unittest.TestCase):

  def test_process(self):
    game_state = GameState(0)

    for i in range(1000):
      bef1 = game_state.s_t[:,:,1]
      bef2 = game_state.s_t[:,:,2]
      bef3 = game_state.s_t[:,:,3]

      game_state.process(1)
      game_state.update()

      aft0 = game_state.s_t[:,:,0]
      aft1 = game_state.s_t[:,:,1]
      aft2 = game_state.s_t[:,:,2]

      # values should be shifted
      self.assertTrue( (bef1.flatten() == aft0.flatten()).all() )
      self.assertTrue( (bef2.flatten() == aft1.flatten()).all() )
      self.assertTrue( (bef3.flatten() == aft2.flatten()).all() )

      # all element should be in [0.0, 1.0]
      for arr in (bef1, bef2, bef3, aft0, aft1, aft2):
        self.assertTrue( np.less_equal(arr, 1.0).all() )
        self.assertTrue( np.greater_equal(arr, 0.0).all() )


if __name__ == '__main__':
  unittest.main()
