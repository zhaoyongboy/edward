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
    _local_name = _name

    class _WrapperRandomVariable(RandomVariable, _candidate):
      def __init__(self, *args, **kwargs):
        RandomVariable.__init__(self, *args, **kwargs)

    _WrapperRandomVariable.__name__ = _local_name
    _globals[_local_name] = _WrapperRandomVariable

    del _WrapperRandomVariable
    del _candidate
    del _local_name


# Rewrite some of the log_probs to expose exponential-family form
def _bernoulli_log_prob(self, value, **kwargs):
  value = tf.cast(value, tf.float32)
  return value * tf.log(self.p) + (1-value) * tf.log(1 - self.p)
Bernoulli.conjugate_log_prob = _bernoulli_log_prob


def _categorical_log_prob(self, value, **kwargs):
  # TODO: This could be very wrong if parameter is not normalized!
  # TODO: Make work with n-d arrays
  value = tf.cast(value, tf.int32)
  return tf.reduce_sum(tf.one_hot(value, tf.shape(self.logits)[-1]) * self.logits, 1)
#   return _categorical_select(self.logits, value)
Categorical.conjugate_log_prob = _categorical_log_prob


class _ParamMixtureDist(distributions.Distribution):
  def __init__(self,
               cat, components,
               validate_args=False,
               allow_nan_stats=True,
               name="ParamMixture"):
    with tf.name_scope(name) as ns:
      with tf.control_dependencies([]):
        self.cat = cat
        self.components = components
        super(_ParamMixtureDist, self).__init__(
            dtype=components.dtype,
            parameters={},
            is_continuous=False,
            is_reparameterized=False,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)


  def sample(self):
    # TODO: Make this more efficient
    cat = self._args[0]
    components = self._args[1]
    selecter = tf.one_hot(cat.value(), tf.shape(components)[-1])
    return tf.reduce_sum(selecter * components.value(), -1)


  def log_prob(self, value, **kwargs):
    log_probs = self.components.conjugate_log_prob(tf.expand_dims(value.value(), -1))
    selecter = tf.one_hot(value.cat.value(), tf.shape(self.components)[-1])
    return tf.reduce_sum(selecter * log_probs, -1)


class ParamMixture(RandomVariable, _ParamMixtureDist):
  def __init__(self, *args, **kwargs):
    super(ParamMixture, self).__init__(*args, **kwargs)
