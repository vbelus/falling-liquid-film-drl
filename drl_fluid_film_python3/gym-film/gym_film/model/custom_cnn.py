import numpy as np
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
import warnings
import param

import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

arch = param.arch
init_scale = np.sqrt(2)


def custom_cnn(scaled_images, **kwargs):
    activ = tf.tanh

    shape = scaled_images.shape

    if len(shape) == 4:
        njets = scaled_images.shape[1]
        filter_width = scaled_images.shape[2]
    if len(shape) == 3:
        njets = 1
        filter_width = scaled_images.shape[1]
        scaled_images = tf.reshape(scaled_images, [-1, njets, filter_width, 2])

    n_filters = arch[0]
    # first conv
    layer = activ(conv(scaled_images, 'c1', n_filters=n_filters,
                       filter_size=(1, filter_width),
                       stride=1,
                       init_scale=init_scale,
                       **kwargs))
    # shape it so that we can apply another conv
    layer = tf.reshape(layer, [-1, njets, n_filters, 1])
    n_filters_prev = n_filters

    for i, n_filters in enumerate(arch):
        # pass first layer
        if i == 0:
            continue

        # apply conv
        layer = activ(conv(layer, 'c'+str(i+1), n_filters=n_filters,
                           filter_size=(1, n_filters_prev),
                           stride=1,
                           init_scale=init_scale,
                           **kwargs))

        # shape it so that we can apply another conv
        layer = tf.reshape(layer, [-1, njets, n_filters, 1])

        # get n_filters of current layer for the next layer
        n_filters_prev = n_filters

    # get back to a flat tensor of size n_jet
    last_layer = conv_to_fc(layer)

    return last_layer


class CustomFeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=None, feature_extraction="cnn", **kwargs):
        super(CustomFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                      scale=False)

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(
                    self.processed_obs, **kwargs)
            else:
                Exception("nope")

            assert str(type(self.pdtype)) == "<class 'stable_baselines.common.distributions.DiagGaussianProbabilityDistributionType'>"

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(
                    pi_latent, vf_latent, init_scale=0.01)

           
            # self._value_fn = linear(vf_latent, 'vf', 1)

            # mean = pi_latent
            # n_jets = mean.shape[1]
            # print("njets", n_jets)
            # logstd = tf.get_variable(name='pi/logstd', shape=[1, n_jets], initializer=tf.zeros_initializer())
            # pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

            # # we take the last layer of our cnn for policy and q value
            # self._proba_distribution = self.pdtype.proba_distribution_from_flat(pdparam)
            # self._policy = pi_latent
            # self.q_value = vf_latent
            # print("heyyyyyyy")

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class CustomCnnPolicy(CustomFeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                              cnn_extractor=custom_cnn, feature_extraction="cnn", **_kwargs)
