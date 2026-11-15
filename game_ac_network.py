# -*- coding: utf-8 -*-
# Defines the actor-critic network architecture shared by the global network
# and every worker thread's local copy. GameACNetwork holds everything that
# doesn't depend on architecture (loss, weight sync, weight init); the two
# subclasses below only differ in how they turn pixels into the pi/v heads.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])

      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

      # policy entropy -- encourages exploration by rewarding a flatter
      # action distribution; scaled by entropy_beta and added to the loss
      entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      # log_pi * a picks out log(pi[taken_action]) per row via the one-hot `a`;
      # multiplying by td (the advantage R-V) is the policy-gradient update rule
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [None])

      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up so a single optimizer
      # step updates both the actor and critic heads together
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()

  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    # This is how "asynchronous" actually works: no locks, no message
    # passing -- a worker just periodically overwrites its local variables
    # with a straight copy of the global network's current values.
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  # (uniform init scaled by 1/sqrt(fan_in), keeping initial activations
  # reasonably scaled regardless of layer size)
  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network (no recurrence -- each frame-stack is judged independently)
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    # each instance (global, or one per thread) gets its own variable scope
    # "net_<thread_index>" so their weights never collide in the same graph
    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2

      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input) -- the 4 stacked frames from GameState.s_t
      self.s = tf.placeholder("float", [None, 84, 84, 4])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592]) # 9*9*32 = 2592
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output) -- probability distribution over actions
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output) -- scalar estimate of the state's expected return
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

# Actor-Critic LSTM Network (adds recurrence so the agent can carry memory
# across frames, e.g. to track things motion-blur alone can't capture)
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2

      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # lstm
      self.lstm = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      # h_fc1 shape=(5,256)

      # dynamic_rnn expects (batch, time, features); here "batch" is always
      # 1 because we're unrolling ONE continuous episode trajectory over
      # time, not a batch of independent sequences
      h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])
      # h_fc_reshaped = (1,5,256)

      # place holder for LSTM unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      # the hidden state the rollout starts from -- carried over from the
      # end of the *previous* rollout (see a3c_training_thread.process)
      self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)

      # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
      # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
      # Unrolling step size is applied via self.step_size placeholder.
      # When forward propagating, step_size is 1.
      # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
      lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False,
                                                        scope = scope)

      # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.

      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)

      # value (output)
      v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

      # dynamic_rnn buries the LSTM's own kernel/bias inside its scope;
      # reuse_variables() + get_variable() pulls them back out by name so
      # get_vars() below can list them for gradients/sync like every other weight
      scope.reuse_variables()
      self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
      self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

      self.reset_state()

  def reset_state(self):
    # zero hidden state -- called at episode boundaries so memory doesn't
    # leak across unrelated episodes
    self.lstm_state_out = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    # The LSTM state advances for real here -- the next call picks up from
    # self.lstm_state_out, which is exactly what we want while acting.
    pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state0 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1 : self.lstm_state_out[1],
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )

    return pi_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    # (Subtlety: unlike run_policy_and_value, this call must NOT permanently
    # advance lstm_state_out -- it's only peeking at V(s_t) for bootstrapping,
    # and the very same s_t will be re-evaluated fresh at the start of the
    # next rollout once the weights have been updated.)
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v, self.lstm_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )

    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]
