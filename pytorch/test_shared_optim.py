# Same scenario as ../rmsprop_applier_test.py, checked against this
# folder's hand-written SharedRMSprop instead of the TF version's
# RMSPropApplier -- applies two gradients in sequence and checks the
# result against the same hand-computed RMSProp update, proving the
# PyTorch port's math matches the original term for term.
import math
import unittest

import torch

from shared_optim import SharedRMSprop


class SharedRMSpropTest(unittest.TestCase):
  def test_apply(self):
    var = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = SharedRMSprop([var], lr=2.0, decay=0.9, momentum=0.0, epsilon=1.0)

    # apply grad0's update
    var.grad = torch.tensor([2.0, 4.0])
    opt.step()

    ms_x = 1.0
    ms_y = 1.0
    x = 1.0
    y = 2.0
    dx = 2.0
    dy = 4.0
    ms_x = ms_x + (dx * dx - ms_x) * (1.0 - 0.9)
    ms_y = ms_y + (dy * dy - ms_y) * (1.0 - 0.9)
    x = x - (2.0 * dx / math.sqrt(ms_x + 1.0))
    y = y - (2.0 * dy / math.sqrt(ms_y + 1.0))

    torch.testing.assert_close(var.detach(), torch.tensor([x, y]), rtol=1e-4, atol=1e-4)

    # apply grad1's update, reusing the same accumulator from grad0
    var.grad = torch.tensor([3.0, 6.0])
    opt.step()

    dx = 3.0
    dy = 6.0
    ms_x = ms_x + (dx * dx - ms_x) * (1.0 - 0.9)
    ms_y = ms_y + (dy * dy - ms_y) * (1.0 - 0.9)
    x = x - (2.0 * dx / math.sqrt(ms_x + 1.0))
    y = y - (2.0 * dy / math.sqrt(ms_y + 1.0))

    torch.testing.assert_close(var.detach(), torch.tensor([x, y]), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
