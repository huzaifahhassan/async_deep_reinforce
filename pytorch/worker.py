# PyTorch counterpart of ../a3c_training_thread.py. One worker process's
# full cycle: sync weights from the global model, play a short rollout,
# compute gradients locally, and push them onto the global model's shared
# parameters through the shared optimizer. train.py spawns PARALLEL_SIZE
# of these as torch.multiprocessing processes, all sharing the same
# global_model and optimizer.
import time

import numpy as np
import torch

from game_state import GameState
from model import GameACFFNetwork, GameACLSTMNetwork, step_loss
from constants import GAMMA, LOCAL_T_MAX, ENTROPY_BETA, ACTION_SIZE, USE_LSTM

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


def _to_tensor(s_t):
  # GameState.s_t is (84, 84, 4) HWC float32 in [0,1]; torch conv layers
  # want (channels, H, W), batched
  return torch.from_numpy(s_t).permute(2, 0, 1).unsqueeze(0)


def anneal_learning_rate(optimizer, initial_lr, global_t, max_global_time_step):
  # linear decay from initial_lr down to 0 as the *shared* global_t
  # approaches max_global_time_step -- mirrors _anneal_learning_rate in
  # a3c_training_thread.py. Setting group['lr'] here is safe across
  # processes: param_groups is plain Python state copied at fork time,
  # not one of the tensors we explicitly put in shared memory, so each
  # worker process has its own independent copy to set.
  lr = initial_lr * (max_global_time_step - global_t) / max_global_time_step
  lr = max(lr, 0.0)
  for group in optimizer.param_groups:
    group['lr'] = lr


def ensure_shared_grads(local_model, global_model):
  # copies this process's freshly-computed local gradients onto the
  # global model's .grad, so the (shared-memory) optimizer step below
  # updates the real global parameters -- the PyTorch analogue of
  # a3c_training_thread.py handing its local gradients to grad_applier,
  # which applies them to global_network.get_vars().
  for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
    global_param._grad = local_param.grad


def worker(rank, global_model, optimizer, counter, counter_lock, stop_flag,
           initial_lr, max_global_time_step, start_time):
  torch.manual_seed(113 * rank)
  np.random.seed(113 * rank)
  # distinct seed per process -> each worker plays a decorrelated game.
  # rank 0 also renders its gameplay live (display=True) so training
  # progress can be watched on screen; the other 7 workers stay headless.
  game_state = GameState(113 * rank, display=(rank == 0))

  if USE_LSTM:
    local_model = GameACLSTMNetwork(ACTION_SIZE)
  else:
    local_model = GameACFFNetwork(ACTION_SIZE)

  if USE_LSTM:
    hx, cx = local_model.init_state()

  episode_reward = 0.0
  local_t = 0
  prev_local_t = 0
  prev_loss_log_t = 0

  while True:
    if stop_flag.value:
      break
    with counter_lock:
      global_t = counter.value
    if global_t > max_global_time_step:
      break

    # copy weights from shared global model to local -- the counterpart
    # of GameACNetwork.sync_from in the TF version
    local_model.load_state_dict(global_model.state_dict())

    actions, rewards, pis, values = [], [], [], []
    terminal_end = False
    start_local_t = local_t

    # play up to LOCAL_T_MAX steps (fewer if the episode ends first)
    # before computing a gradient update. Gradients are NOT disabled here
    # (unlike the run_policy_and_value() convenience wrapper on the model
    # classes) because the pi/v tensors produced during this loop are
    # reused directly below to build the loss -- for the LSTM network in
    # particular, chaining nn.LSTMCell calls through this python loop
    # with grad enabled is exactly what gives autograd the full
    # backprop-through-time graph, the same thing tf.nn.dynamic_rnn's
    # single call over a stored sequence achieves in the TF version.
    for i in range(LOCAL_T_MAX):
      s_t = _to_tensor(game_state.s_t)

      if USE_LSTM:
        pi, v, (hx, cx) = local_model(s_t, hx, cx)
      else:
        pi, v = local_model(s_t)
      pi, v = pi[0], v[0]

      action = np.random.choice(ACTION_SIZE, p=pi.detach().numpy())

      actions.append(action)
      pis.append(pi)
      values.append(v)

      if rank == 0 and local_t % LOG_INTERVAL == 0:
        print("pi={}".format(pi.detach().numpy()))
        print(" V={}".format(v.item()))

      game_state.process(action)
      reward = game_state.reward
      terminal = game_state.terminal
      episode_reward += reward

      # clip reward to [-1, 1] so one game's reward scale doesn't dominate
      # the value loss relative to another's
      rewards.append(np.clip(reward, -1, 1))

      local_t += 1
      game_state.update()

      if terminal:
        terminal_end = True
        print("worker {} score={}".format(rank, episode_reward))
        episode_reward = 0.0
        game_state.reset()
        if USE_LSTM:
          hx, cx = local_model.init_state() # fresh episode -> fresh memory
        break

    R = 0.0
    if not terminal_end:
      # rollout ended mid-episode: bootstrap the remaining return with the
      # critic's own value estimate of the final state instead of 0.
      # This peek must NOT permanently move hx/cx forward (hx, cx below
      # are intentionally not reassigned) -- the next rollout's first
      # real action-selection call has to start from the same hidden
      # state this peek used as input, not from what it produced. See
      # the matching comment on run_value in ../game_ac_network.py.
      s_t = _to_tensor(game_state.s_t)
      with torch.no_grad():
        if USE_LSTM:
          _, v_final, _ = local_model(s_t, hx, cx)
        else:
          _, v_final = local_model(s_t)
      R = v_final[0].item()

    # walk the rollout backwards, accumulating the discounted n-step
    # return R (a plain float -- like self.r in the TF version, it's a
    # constant target, never something we backprop through) and the
    # resulting advantage (R - V) at each step, exactly like the backward
    # loop in a3c_training_thread.py.process()
    total_loss = torch.zeros(())
    for i in reversed(range(len(rewards))):
      R = rewards[i] + GAMMA * R
      advantage = (R - values[i]).detach()
      total_loss = total_loss + step_loss(pis[i], values[i], actions[i],
                                          advantage, R, ENTROPY_BETA)

    # boundary-crossing check, not an exact modulo -- a rollout can end
    # early (episode terminates mid-rollout), which shifts local_t off of
    # exact multiples of LOG_INTERVAL permanently; "did we cross a
    # LOG_INTERVAL boundary since the last print" still fires reliably
    # regardless of that drift (same pattern as the PERFORMANCE_LOG_INTERVAL
    # check below, for the same reason).
    if rank == 0 and local_t - prev_loss_log_t >= LOG_INTERVAL:
      prev_loss_log_t = local_t
      print("worker 0 loss={:.4f} (avg/step={:.4f}) at local_t={}".format(
          total_loss.item(), total_loss.item() / len(rewards), local_t))

    anneal_learning_rate(optimizer, initial_lr, global_t, max_global_time_step)

    local_model.zero_grad()
    total_loss.backward()
    ensure_shared_grads(local_model, global_model)
    optimizer.step()

    if USE_LSTM:
      # detach the carried-over hidden state from this rollout's graph --
      # otherwise the next rollout's backward pass would try to walk all
      # the way back through every previous rollout too
      hx, cx = hx.detach(), cx.detach()

    with counter_lock:
      counter.value += (local_t - start_local_t)

    if rank == 0 and local_t - prev_local_t >= PERFORMANCE_LOG_INTERVAL:
      prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - start_time
      steps_per_sec = global_t / elapsed_time if elapsed_time > 0 else 0.0
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
