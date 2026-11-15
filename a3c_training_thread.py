# -*- coding: utf-8 -*-
# One worker's full cycle: sync weights from the global network, play a
# short rollout, compute gradients locally, and push them onto the global
# network's variables. a3c.py creates PARALLEL_SIZE of these, one per
# Python thread, all sharing the same global_network and grad_applier.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
import time
import sys

from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    # each thread gets its own local network instance (own variable scope,
    # see game_ac_network.py) so it can run forward/backward passes without
    # fighting the other threads over the same tensors
    if USE_LSTM:
      self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
    else:
      self.local_network = GameACFFNetwork(ACTION_SIZE, thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      # var_refs via ._ref() is the TF1 idiom for taking gradients against
      # the mutable Variable itself rather than a snapshot of its value
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    # this is the actual "asynchronous" wiring: gradients are computed from
    # the LOCAL network's loss, but applied to the GLOBAL network's
    # variables. grad_applier is the one shared RMSPropApplier instance
    # created once in a3c.py, so its rms/momentum slots are shared by every
    # thread's updates too -- no per-thread import of rmsprop_applier needed,
    # this object is just handed in and called.
    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )

    # op run at the start of every rollout to pull the latest global weights
    # into this thread's local network (see GameACNetwork.sync_from)
    self.sync = self.local_network.sync_from(global_network)

    # distinct seed per thread -> each worker plays a decorrelated game
    self.game_state = GameState(113 * thread_index)

    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  def _anneal_learning_rate(self, global_time_step):
    # linear decay from initial_learning_rate down to 0 as the *shared*
    # global_time_step approaches max_global_time_step
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    # sample an action from the policy's own probability distribution --
    # this sampling *is* the exploration, no extra epsilon-greedy needed
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if USE_LSTM:
      # remember the LSTM hidden state as it stood at the *start* of this
      # rollout -- needed below to replay the same history during backprop
      start_lstm_state = self.local_network.lstm_state_out

    # t_max times loop -- play up to LOCAL_T_MAX steps (fewer if the
    # episode ends first) before computing a gradient update
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      action = self.choose_action(pi_)

      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward to [-1, 1] so one game's reward scale doesn't dominate
      # the value loss relative to another's
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()

      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward))

        self._record_score(sess, summary_writer, summary_op, score_input,
                           self.episode_reward, global_t)

        self.episode_reward = 0
        self.game_state.reset()
        if USE_LSTM:
          self.local_network.reset_state() # fresh episode -> fresh memory
        break

    R = 0.0
    if not terminal_end:
      # rollout ended mid-episode: bootstrap the remaining return with the
      # critic's own value estimate of the final state instead of 0
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    # walking backward from the last step, accumulate the discounted
    # n-step return R, then use it to get the advantage td = R - V for
    # the policy gradient (and R itself as the critic's regression target)
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if USE_LSTM:
      # un-reverse back into chronological order so dynamic_rnn unrolls
      # the sequence forwards, continuing from start_lstm_state
      batch_si.reverse()
      batch_a.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.learning_rate_input: cur_learning_rate} )

    # throughput logging, only from thread 0 to avoid duplicate/garbled prints
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size -- a3c.py adds this to the shared
    # global_t counter (not atomically; small race conditions here are an
    # accepted, lock-free tradeoff of the async design)
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

