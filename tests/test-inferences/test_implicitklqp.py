from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal
from tensorflow.contrib import slim


class test_implicit_klqp_class(tf.test.TestCase):

  def test_normal_run(self):
    def ratio_estimator(data, local_vars, global_vars):
      input = tf.reshape(local_vars[z], [1, 1])  # reshape scalar as matrix input
      h1 = slim.fully_connected(input, 10, activation_fn=tf.nn.relu)
      h2 = slim.fully_connected(h1, 1, activation_fn=None)
      return h2


    with self.test_session() as sess:
      z = Normal(mu=5.0, sigma=1.0)

      qz = Normal(mu=tf.Variable(tf.random_normal([])),
                  sigma=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

      inference = ed.ImplicitKLqp({z: qz}, discriminator=ratio_estimator)
      # inference.run(n_iter=1000)
      inference.initialize(n_iter=1000, n_print=100)

      sess = ed.get_session()
      tf.global_variables_initializer().run()

      for _ in range(inference.n_iter):
        info_dict = inference.update()
        t = info_dict['t']
        inference.print_progress(info_dict)
        if t == 1 or t % inference.n_print == 0:
          # Check inferred posterior parameters.
          mean, std = sess.run([qz.mean(), qz.std()])
          print("Inferred mean & std: {} {}".format(mean, std))

      self.assertAllClose(qz.mean().eval(), 5.0, atol=1.0)

if __name__ == '__main__':
  ed.set_seed(47324)
  tf.test.main()
