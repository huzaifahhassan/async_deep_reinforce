# PyTorch counterpart of ../a3c_display.py. Inference-only: loads a
# checkpoint and watches the trained agent play Pong live
# (GameState(display=True)). No training happens here.
import os

import numpy as np
import torch

from game_state import GameState
from model import GameACFFNetwork, GameACLSTMNetwork

from constants import ACTION_SIZE, CHECKPOINT_DIR, USE_LSTM
from train import _find_latest_checkpoint


def choose_action(pi_values):
  # sample from the policy distribution, same as during training
  return np.random.choice(range(len(pi_values)), p=pi_values)


def main():
  if USE_LSTM:
    model = GameACLSTMNetwork(ACTION_SIZE)
  else:
    model = GameACFFNetwork(ACTION_SIZE)

  checkpoint_path = _find_latest_checkpoint()
  if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print("checkpoint loaded:", checkpoint_path)
  else:
    print("Could not find old checkpoint")
  model.eval()

  game_state = GameState(0, display=True, no_op_max=0)

  # forever loop: query the policy, act, repeat -- rendered on screen
  # because display=True was passed above
  while True:
    s_t = torch.from_numpy(game_state.s_t).permute(2, 0, 1)
    pi_values = model.run_policy(s_t).numpy()

    action = choose_action(pi_values)
    game_state.process(action)

    if game_state.terminal:
      game_state.reset()
      if USE_LSTM:
        model.reset_state()
    else:
      game_state.update()


if __name__ == "__main__":
  main()
