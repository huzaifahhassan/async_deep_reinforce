# -*- coding: utf-8 -*-
# Training entry point. Builds one global network + one shared optimizer,
# spins up PARALLEL_SIZE worker threads (see a3c_training_thread.py) that
# all read/write those shared objects concurrently, and checkpoints the
# result. Run this file directly to train; nothing else imports it.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import threading
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM


def log_uniform(lo, hi, rate):
  # samples a value log-uniformly between lo and hi; rate=0 -> lo, rate=1 -> hi.
  # Used once below to pick a random learning rate per run, per the A3C paper.
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0 # step counter shared (non-atomically) across all worker threads

stop_requested = False

# thread_index = -1 marks this as the global network (vs. 0..PARALLEL_SIZE-1
# for each worker's local copy) -- just used for the variable_scope name
if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  global_network = GameACFFNetwork(ACTION_SIZE, -1, device)


training_threads = []

learning_rate_input = tf.placeholder("float")

# one optimizer instance shared by every worker thread -- this is the
# "shared RMSProp" from the A3C paper: one set of rms/momentum statistics
# per global variable, not one optimizer per thread
grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(PARALLEL_SIZE):
  # constructing a training_thread only builds its graph nodes (local
  # network, gradients, sync op) -- no session exists yet, so this is safe
  # to do before sess.run(init) below
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP,
                                      device = device)
  training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  # checkpoint filenames are "checkpoint-<global_t>" -- recover the step
  # count so training resumes where it left off instead of restarting at 0
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time -- restore how much wall-clock time had already elapsed,
  # so steps/sec performance logging stays accurate across restarts
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0


def train_function(parallel_index):
  # this is the body each worker Thread runs: repeatedly play a rollout and
  # apply its gradients until told to stop or the step budget runs out
  global global_t

  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  training_thread.set_start_time(start_time)

  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                            summary_op, score_input)
    global_t += diff_global_t


def signal_handler(signal, frame):
  # Ctrl+C requests a graceful stop instead of killing the process outright,
  # so the checkpoint-save code below still gets to run
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True

train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause() # main thread idles here until SIGINT fires signal_handler

print('Now saving data. Please wait')

for t in train_threads:
  t.join()

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
  f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

