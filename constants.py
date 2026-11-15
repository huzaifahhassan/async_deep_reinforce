# -*- coding: utf-8 -*-
# Every tunable value for the A3C Pong agent lives here, so every other file
# imports from this single source of truth instead of hard-coding numbers.

LOCAL_T_MAX = 20 # steps a worker plays before it bootstraps + syncs (the "n" in n-step returns)
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 8 # number of asynchronous worker threads (each owns one GameState + one local network)
ROM = "pong.bin"     # Atari ROM to load
ACTION_SIZE = 3 # size of Pong's minimal action set -- must match what the ROM actually exposes

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 10 * 10**7 # training stops once the shared global step counter passes this
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
USE_LSTM = False # True for A3C LSTM, False for A3C FF -- picks which GameACNetwork subclass every file builds
