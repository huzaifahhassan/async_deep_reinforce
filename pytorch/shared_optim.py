# PyTorch counterpart of ../rmsprop_applier.py.
#
# Why not just use torch.optim.RMSprop? Because A3C needs "shared" RMSProp:
# one set of accumulator statistics per *global* parameter, updated by
# whichever worker process finishes a rollout, not one optimizer per
# process with its own private statistics. torch.multiprocessing forks
# each worker from the same parent, and a plain optimizer's state tensors
# would get copied at fork time just like anything else in memory -- each
# process would end up quietly optimizing against its own stale copy of
# the moving averages. Calling .share_memory_() on the state tensors here,
# before any process forks, keeps them backed by the same shared-memory
# block everywhere, so every worker's step() reads and writes the same
# numbers -- the exact role rmsprop_applier.py's per-variable "rms" and
# "momentum" slots play for the TF threads.
#
# The update rule itself is written out by hand (rather than subclassing
# torch.optim.RMSprop) so it matches rmsprop_applier.py's math exactly,
# term for term -- see test_shared_optim.py, which checks it against the
# same hand-computed numbers as ../rmsprop_applier_test.py.
import torch
from torch.optim import Optimizer


class SharedRMSprop(Optimizer):
  def __init__(self, params, lr, decay=0.99, momentum=0.0, epsilon=0.1,
               clip_norm=40.0):
    # clip_norm lives in `defaults` (so it ends up in each param_group)
    # rather than as a plain self.clip_norm attribute. Optimizer defines
    # its own __getstate__ that only pickles {defaults, state,
    # param_groups} -- a bare instance attribute would silently vanish
    # whenever this optimizer gets pickled, which torch.multiprocessing's
    # "spawn" start method (required on Windows, since there's no fork())
    # does to every argument passed to a child process.
    defaults = dict(lr=lr, decay=decay, momentum=momentum, epsilon=epsilon,
                     clip_norm=clip_norm)
    super().__init__(params, defaults)

    # _create_slots equivalent: one "square_avg" (the "rms" slot) and one
    # "momentum_buffer" per parameter, created now so they exist (and can
    # be moved into shared memory) before any worker process forks
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        # matches rmsprop_applier.py's _create_slots, which seeds "rms"
        # with a constant 1.0 tensor, not zeros
        state['square_avg'] = torch.ones_like(p.data)
        state['momentum_buffer'] = torch.zeros_like(p.data)

    self.share_memory()

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['square_avg'].share_memory_()
        state['momentum_buffer'].share_memory_()

  @torch.no_grad()
  def step(self, closure=None):
    for group in self.param_groups:
      # clip each gradient's norm before applying it -- several worker
      # processes are hitting these same global parameters concurrently,
      # and one unstable gradient from any of them would otherwise be
      # able to blow up the shared weights (same reasoning as
      # tf.clip_by_norm in rmsprop_applier.py's apply_gradients)
      torch.nn.utils.clip_grad_norm_(group['params'], group['clip_norm'])

      lr = group['lr']
      decay = group['decay']
      momentum = group['momentum']
      epsilon = group['epsilon']

      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        state = self.state[p]
        square_avg = state['square_avg']
        mom = state['momentum_buffer']

        # ms <- decay * ms + (1 - decay) * grad^2
        square_avg.mul_(decay).addcmul_(grad, grad, value=1 - decay)
        # mom <- momentum * mom + lr * grad / sqrt(ms + epsilon)
        avg = square_avg.add(epsilon).sqrt_()
        mom.mul_(momentum).addcdiv_(grad, avg, value=lr)
        # var <- var - mom
        p.data.add_(mom, alpha=-1)

    return None
