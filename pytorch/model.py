# PyTorch counterpart of ../game_ac_network.py. Same two architectures
# (feed-forward and LSTM actor-critic), same fan-in-scaled weight init, and
# the same loss, just expressed as nn.Module + plain Python functions
# instead of a hand-built TensorFlow graph.
#
# There's no sync_from() here: in the TF version that op exists because the
# global network and each thread's local network are separate sets of TF
# variables that must be copied explicitly. Here, workers instead keep a
# LOCAL nn.Module and call local_model.load_state_dict(global_model.state_dict())
# at the start of every rollout (see worker.py) -- same idea, PyTorch's own
# idiom for it.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(layer):
  # mirrors _conv_variable in game_ac_network.py: uniform init scaled by
  # 1/sqrt(fan_in), based on muupan's async-rl code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
  d = 1.0 / math.sqrt(fan_in)
  nn.init.uniform_(layer.weight, -d, d)
  nn.init.uniform_(layer.bias, -d, d)


def _init_fc(layer):
  # mirrors _fc_variable in game_ac_network.py
  fan_in = layer.in_features
  d = 1.0 / math.sqrt(fan_in)
  nn.init.uniform_(layer.weight, -d, d)
  nn.init.uniform_(layer.bias, -d, d)


class _ConvTrunk(nn.Module):
  """The conv+fc stack shared by both network variants below -- the part
  before the policy/value heads split off."""

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)  # stride=4
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # stride=2
    self.fc1 = nn.Linear(2592, 256)  # 9*9*32 = 2592, same flatten size as the TF version
    _init_conv(self.conv1)
    _init_conv(self.conv2)
    _init_fc(self.fc1)

  def forward(self, s):
    # s: (batch, 4, 84, 84) -- the 4 stacked frames from GameState.s_t
    x = F.relu(self.conv1(s))
    x = F.relu(self.conv2(x))
    x = x.reshape(x.size(0), -1)
    x = F.relu(self.fc1(x))
    return x


class GameACFFNetwork(nn.Module):
  """Actor-Critic FF Network -- no recurrence, each frame-stack judged independently."""

  def __init__(self, action_size):
    super().__init__()
    self.trunk = _ConvTrunk()
    self.fc_pi = nn.Linear(256, action_size)
    self.fc_v = nn.Linear(256, 1)
    _init_fc(self.fc_pi)
    _init_fc(self.fc_v)

  def forward(self, s):
    h = self.trunk(s)
    pi = F.softmax(self.fc_pi(h), dim=-1)
    v = self.fc_v(h).view(-1)
    return pi, v

  def run_policy_and_value(self, s_t):
    with torch.no_grad():
      pi, v = self.forward(s_t.unsqueeze(0))
    return pi[0], v[0]

  def run_policy(self, s_t):
    with torch.no_grad():
      pi, _ = self.forward(s_t.unsqueeze(0))
    return pi[0]

  def run_value(self, s_t):
    with torch.no_grad():
      _, v = self.forward(s_t.unsqueeze(0))
    return v[0]


class GameACLSTMNetwork(nn.Module):
  """Actor-Critic LSTM Network -- adds recurrence so the agent can carry
  memory across frames. Uses nn.LSTMCell stepped once per frame in a plain
  Python loop (see worker.py) instead of tf.nn.dynamic_rnn's single call
  over a whole sequence -- PyTorch's autograd tracks the chain through the
  loop automatically, so backprop-through-time falls out for free."""

  def __init__(self, action_size):
    super().__init__()
    self.trunk = _ConvTrunk()
    self.lstm = nn.LSTMCell(256, 256)
    self.fc_pi = nn.Linear(256, action_size)
    self.fc_v = nn.Linear(256, 1)
    _init_fc(self.fc_pi)
    _init_fc(self.fc_v)
    # the LSTMCell's own weights are left at PyTorch's default init -- the
    # TF version doesn't have a direct counterpart for it either (BasicLSTMCell
    # uses its own internal default there too)
    self.reset_state()

  def forward(self, s, hx, cx):
    h = self.trunk(s)
    hx, cx = self.lstm(h, (hx, cx))
    pi = F.softmax(self.fc_pi(hx), dim=-1)
    v = self.fc_v(hx).view(-1)
    return pi, v, (hx, cx)

  def init_state(self, batch_size=1, device=None):
    return (torch.zeros(batch_size, 256, device=device),
            torch.zeros(batch_size, 256, device=device))

  def reset_state(self, device=None):
    # zero hidden state -- call at episode boundaries so memory doesn't
    # leak across unrelated episodes
    self.lstm_state_out = self.init_state(device=device)

  # --- inference-only convenience wrappers, mirroring the TF version's
  # stateful run_policy_and_value/run_policy/run_value (they manage
  # self.lstm_state_out internally instead of making the caller thread
  # hx/cx through by hand). worker.py's training loop does NOT use these --
  # it needs gradients to flow, so it calls forward() directly and threads
  # hx/cx itself. These are for display.py / visualize.py style pure
  # inference instead. ---

  def run_policy_and_value(self, s_t):
    with torch.no_grad():
      pi, v, self.lstm_state_out = self.forward(s_t.unsqueeze(0), *self.lstm_state_out)
    return pi[0], v[0]

  def run_policy(self, s_t):
    with torch.no_grad():
      pi, _, self.lstm_state_out = self.forward(s_t.unsqueeze(0), *self.lstm_state_out)
    return pi[0]

  def run_value(self, s_t):
    # peek at V(s_t) without permanently advancing the recurrent state --
    # see the matching comment on run_value in ../game_ac_network.py
    prev_state = self.lstm_state_out
    with torch.no_grad():
      _, v, _ = self.forward(s_t.unsqueeze(0), *self.lstm_state_out)
    self.lstm_state_out = prev_state
    return v[0]


def l2_loss(x):
  """Direct port of tf.nn.l2_loss: 0.5 * sum(x**2)."""
  return 0.5 * (x ** 2).sum()


def step_loss(pi, v, action_index, advantage, target_return, entropy_beta):
  """
  One rollout step's contribution to the total A3C loss -- the PyTorch
  counterpart of GameACNetwork.prepare_loss, evaluated per step instead of
  as one TF graph fed by whole-batch placeholders. Call this once per
  stored (pi, v) pair while walking a finished rollout backwards and
  accumulating n-step returns (see worker.py), summing the results into
  the one loss tensor you call .backward() on.

  `advantage` (= target_return - v, computed by the caller) must already
  be detached: it's meant to be a constant coefficient for the policy
  gradient, not something v's own gradient should flow back through --
  this mirrors self.td being a *fed* placeholder in the TF version, never
  a graph node connected to self.v. `target_return` (R) must likewise be
  a detached constant -- the value loss regresses v towards it, but R
  itself carries no gradient, same as self.r there.
  """
  # avoid NaN with clipping when a value in pi becomes zero
  log_pi = torch.log(torch.clamp(pi, 1e-20, 1.0))

  # policy entropy -- encourages exploration by rewarding a flatter
  # action distribution; scaled by entropy_beta and added to the loss
  entropy = -(pi * log_pi).sum()

  # policy loss (negated because the paper's objective is for gradient
  # ascent, but torch optimizers do gradient descent)
  policy_loss = -(log_pi[action_index] * advantage + entropy_beta * entropy)

  # value loss (learning rate for the critic is half of the actor's, so
  # the extra 0.5 here on top of l2_loss's own built-in 1/2)
  value_loss = 0.5 * l2_loss(target_return - v)

  return policy_loss + value_loss
