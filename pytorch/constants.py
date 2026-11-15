# Same hyperparameters as the TensorFlow version one directory up
# (../constants.py), duplicated here so this folder is self-contained and
# importable on its own.

LOCAL_T_MAX = 20 # steps a worker plays before it bootstraps + syncs (the "n" in n-step returns)
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints_pytorch'
LOG_DIR = 'tmp/a3c_log_pytorch'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 8 # number of worker processes (each owns one GameState + one local model)
ROM = "pong.bin"     # Atari ROM, lives at the repo root (see game_state.py for path resolution)
ACTION_SIZE = 3 # size of Pong's minimal action set -- must match what the ROM actually exposes

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regularization constant
MAX_TIME_STEP = 10 * 10**7 # training stops once the shared global step counter passes this
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # set True to run forward/backward passes on CUDA
USE_LSTM = False # True for A3C LSTM, False for A3C FF -- picks which model class every file builds
