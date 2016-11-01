from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tensorflow as tf

from edward.models.empirical import Empirical as distributions_Empirical
from edward.models.point_mass import PointMass as distributions_PointMass
from edward.models.random_variable import RandomVariable
from tensorflow.contrib import distributions


class Empirical(RandomVariable, distributions_Empirical):
  def __init__(self, *args, **kwargs):
    super(Empirical, self).__init__(*args, **kwargs)


class PointMass(RandomVariable, distributions_PointMass):
  def __init__(self, *args, **kwargs):
    super(PointMass, self).__init__(*args, **kwargs)


# Automatically generate random variable classes from classes in
# tf.contrib.distributions.
_globals = globals()
for _name in sorted(dir(distributions)):
  _candidate = getattr(distributions, _name)
  if (inspect.isclass(_candidate) and
          _candidate != distributions.Distribution and
          issubclass(_candidate, distributions.Distribution)):

    class _WrapperRandomVariable(RandomVariable, _candidate):
      def __init__(self, *args, **kwargs):
        RandomVariable.__init__(self, *args, **kwargs)

      def conjugate_log_prob(self, *args, **kwargs):
        # Version of log_prob() in clearer exponential-family form, if needed
        super_obj = super(_globals[type(self).__name__], self)
        return super_obj.log_prob(*args, **kwargs)

    _WrapperRandomVariable.__name__ = _name
    _globals[_name] = _WrapperRandomVariable

    del _WrapperRandomVariable
    del _candidate

# Rewrite some of the log_probs to expose exponential-family form
def _bernoulli_log_prob(self, value, **kwargs):
  value = tf.cast(value, tf.float32)
  return value * tf.log(self.p) + (1-value) * tf.log(1 - self.p)
Bernoulli.conjugate_log_prob = _bernoulli_log_prob

# def _multinomial_log_prob(self, value, **kwargs):
