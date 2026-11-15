# PyTorch counterpart of ../a3c.py. Builds one global model + one shared
# optimizer, spawns PARALLEL_SIZE worker *processes* (see worker.py) that
# all read/write those shared objects concurrently, and checkpoints the
# result. Run this file directly to train.
#
# Unlike a3c.py, the process-spawning code below has to live inside
# main() behind the `if __name__ == "__main__":` guard. On Windows there's
# no fork(): torch.multiprocessing uses the "spawn" start method, which
# re-imports this file fresh in every child process. Without the guard,
# each child would re-execute the top-level spawn code itself and spawn
# its own children recursively.
import glob
import math
import os
import re
import signal
import time

import torch
import torch.multiprocessing as mp

from model import GameACFFNetwork, GameACLSTMNetwork
from shared_optim import SharedRMSprop
from worker import worker

from constants import (ACTION_SIZE, PARALLEL_SIZE, INITIAL_ALPHA_LOW,
                        INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE,
                        MAX_TIME_STEP, CHECKPOINT_DIR, RMSP_EPSILON,
                        RMSP_ALPHA, GRAD_NORM_CLIP, USE_LSTM)


def log_uniform(lo, hi, rate):
  # samples a value log-uniformly between lo and hi; rate=0 -> lo, rate=1 -> hi.
  # Used once below to pick a random learning rate per run, per the A3C paper.
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1 - rate) + log_hi * rate
  return math.exp(v)


def _checkpoint_path(global_t):
  return os.path.join(CHECKPOINT_DIR, "checkpoint-{}.pt".format(global_t))


def _find_latest_checkpoint():
  # plain torch.save files instead of TF's Saver + checkpoint-index file;
  # "latest" is just whichever checkpoint-<global_t>.pt has the highest
  # global_t embedded in its filename
  if not os.path.isdir(CHECKPOINT_DIR):
    return None
  candidates = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*.pt"))
  if not candidates:
    return None
  def step_of(path):
    m = re.search(r"checkpoint-(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1
  return max(candidates, key=step_of)


def main():
  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH,
                                      INITIAL_ALPHA_LOG_RATE)

  if USE_LSTM:
    global_model = GameACLSTMNetwork(ACTION_SIZE)
  else:
    global_model = GameACFFNetwork(ACTION_SIZE)
  global_model.share_memory() # backs every parameter's storage with shared memory

  optimizer = SharedRMSprop(global_model.parameters(),
                            lr=initial_learning_rate,
                            decay=RMSP_ALPHA,
                            momentum=0.0,
                            epsilon=RMSP_EPSILON,
                            clip_norm=GRAD_NORM_CLIP)

  checkpoint_path = _find_latest_checkpoint()
  if checkpoint_path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    global_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    optimizer.share_memory() # re-share the tensors just loaded from disk
    global_t = ckpt["global_t"]
    wall_t = ckpt["wall_t"]
    print("checkpoint loaded:", checkpoint_path)
    print(">>> global step set:", global_t)
  else:
    print("Could not find old checkpoint")
    global_t = 0
    wall_t = 0.0

  # shared across worker processes: a step counter (protected by a lock,
  # since real OS processes -- unlike the TF version's Python threads --
  # have no GIL to make a lock-free += even approximately safe), and a
  # stop flag Ctrl+C can set to request a graceful shutdown
  counter = mp.Value('l', global_t)
  counter_lock = mp.Lock()
  stop_flag = mp.Value('b', False)

  start_time = time.time() - wall_t

  processes = []
  for rank in range(PARALLEL_SIZE):
    p = mp.Process(target=worker, args=(rank, global_model, optimizer,
                                        counter, counter_lock, stop_flag,
                                        initial_learning_rate, MAX_TIME_STEP,
                                        start_time))
    p.start()
    processes.append(p)

  def signal_handler(signum, frame):
    # Ctrl+C requests a graceful stop instead of killing processes outright,
    # so the checkpoint-save code below still gets to run
    print('You pressed Ctrl+C!')
    stop_flag.value = True

  signal.signal(signal.SIGINT, signal_handler)

  print('Press Ctrl+C to stop')
  for p in processes:
    p.join()

  print('Now saving data. Please wait')

  if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

  final_global_t = counter.value
  wall_t = time.time() - start_time
  torch.save({
    "model": global_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "global_t": final_global_t,
    "wall_t": wall_t,
  }, _checkpoint_path(final_global_t))


if __name__ == "__main__":
  main()
