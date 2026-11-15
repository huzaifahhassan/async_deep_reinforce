# Checks RMSPropApplier.apply_gradients() against a hand-computed RMSProp
# update: applies two different gradients in sequence and verifies the
# variable's value matches the textbook formula each time (moving-average
# of squared gradients, then a step scaled by 1/sqrt(ms + epsilon)).
import numpy as np
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import rmsprop_applier

class RMSPropApplierTest(tf.test.TestCase):
  def testApply(self):
    with self.test_session():
      var = tf.Variable([1.0, 2.0])

      grad0 = tf.Variable([2.0, 4.0])
      grad1 = tf.Variable([3.0, 6.0])

      opt = rmsprop_applier.RMSPropApplier(learning_rate=2.0,
                                           decay=0.9,
                                           momentum=0.0,
                                           epsilon=1.0)

      apply_gradient0 = opt.apply_gradients([var], [grad0])
      apply_gradient1 = opt.apply_gradients([var], [grad1])

      tf.initialize_all_variables().run()

      # grad0を反映 (apply grad0's update)
      apply_gradient0.run()

      ms_x = 1.0
      ms_y = 1.0

      x = 1.0
      y = 2.0
      dx = 2.0
      dy = 4.0
      ms_x = ms_x + (dx * dx - ms_x) * (1.0 - 0.9)
      ms_y = ms_y + (dy * dy - ms_y) * (1.0 - 0.9)
      x = x - (2.0 * dx / math.sqrt(ms_x+1.0))
      y = y - (2.0 * dy / math.sqrt(ms_y+1.0))

      self.assertAllClose(np.array([x, y]), var.eval())

      # grad1を反映 (apply grad1's update, reusing the same rms slot from grad0)
      apply_gradient1.run()

      dx = 3.0
      dy = 6.0
      ms_x = ms_x + (dx * dx - ms_x) * (1.0 - 0.9)
      ms_y = ms_y + (dy * dy - ms_y) * (1.0 - 0.9)
      x = x - (2.0 * dx / math.sqrt(ms_x+1.0))
      y = y - (2.0 * dy / math.sqrt(ms_y+1.0))

      self.assertAllClose(np.array([x, y]), var.eval())

if __name__ == "__main__":
  tf.test.main()
