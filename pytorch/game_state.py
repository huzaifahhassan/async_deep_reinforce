# Thin wrapper around the Arcade Learning Environment (ALE) that turns raw
# Atari frames into the fixed-size, normalized tensors the network expects.
# Identical logic to ../game_state.py -- it has no TensorFlow dependency,
# so this is effectively a straight copy, just with a ROM path that's
# resolved relative to the repo root instead of assuming a particular CWD.
import os
import sys
import numpy as np
import cv2
from ale_py import ALEInterface, Action

from constants import ROM

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_ROM_PATH = os.path.join(_REPO_ROOT, ROM)

class GameState(object):
  def __init__(self, rand_seed, display=False, no_op_max=7):
    self.ale = ALEInterface()
    self.ale.setInt('random_seed', rand_seed) # distinct seed per worker -> decorrelated trajectories across processes
    self.ale.setFloat('repeat_action_probability', 0.0) # keep the env deterministic given an action sequence
    self.ale.setBool('color_averaging', True) # blends consecutive frames to avoid flicker artifacts
    self.ale.setInt('frame_skip', 4) # agent only acts every 4th frame; ALE repeats the chosen action in between
    self._no_op_max = no_op_max

    if display:
      self._setup_display()

    self.ale.loadROM(_ROM_PATH)

    # the 3 actions this project trains over: stand still, or move the
    # paddle one way or the other. Deliberately NOT
    # self.ale.getMinimalActionSet()[:3] -- that set's composition isn't
    # stable across ALE versions (it has included FIRE/RIGHTFIRE/LEFTFIRE
    # variants alongside NOOP/RIGHT/LEFT), so slicing it positionally can
    # silently drop one of the two paddle directions instead of FIRE.
    self.real_actions = [Action.NOOP, Action.RIGHT, Action.LEFT]

    # height=210, width=160
    self._screen = np.empty((210, 160), dtype=np.uint8)

    self.reset()

  def _process_frame(self, action, reshape):
    reward = self.ale.act(action)
    terminal = self.ale.game_over()

    # screen shape is (210, 160)
    self.ale.getScreenGrayscale(self._screen)

    # resize to height=110, width=84
    resized_screen = cv2.resize(self._screen, (84, 110))

    # crop the top/bottom letterboxing off the resized frame, leaving an
    # 84x84 square that's just the playfield (no score bar)
    x_t = resized_screen[18:102,:]
    if reshape:
      x_t = np.reshape(x_t, (84, 84, 1))
    x_t = x_t.astype(np.float32)
    x_t *= (1.0/255.0) # normalize pixel values to [0, 1]
    return reward, terminal, x_t


  def _setup_display(self):
    if sys.platform == 'darwin':
      import pygame
      pygame.init()
      self.ale.setBool('sound', False)
    elif sys.platform.startswith('linux'):
      self.ale.setBool('sound', True)
    self.ale.setBool('display_screen', True)

  def reset(self):
    self.ale.reset_game()

    # randomize initial state by taking a random number of no-op actions,
    # so every episode (and every parallel worker) starts from a slightly
    # different point instead of the exact same deterministic opening
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max + 1)
      for _ in range(no_op):
        self.ale.act(0)

    _, _, x_t = self._process_frame(0, False)

    self.reward = 0
    self.terminal = False
    # a single frame can't convey motion (e.g. ball direction/speed), so the
    # network sees the last 4 frames stacked as channels; at reset there's
    # only one real frame, so it's duplicated to fill the stack
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

  def process(self, action):
    # convert the network's action index (0..ACTION_SIZE-1) to ALE's actual
    # action id from the minimal action set
    real_action = self.real_actions[action]

    r, t, x_t1 = self._process_frame(real_action, True)

    self.reward = r
    self.terminal = t
    # build the *next* state by dropping the oldest frame and appending the
    # new one, but don't commit it to s_t yet -- that happens in update().
    self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)

  def update(self):
    self.s_t = self.s_t1
