import param
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv_to_fc, ortho_init
import warnings
import numpy as np
import tensorflow as tf
from itertools import zip_longest

import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0, reuse=False):
    """
    Creates a fully connected layer for TensorFlow
    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable(
            "w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.get_variable(
            "b", [n_hidden], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias


def mlp_extractor(observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].
    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """

    shape = observations.shape
    print(shape)

    if len(shape) == 4:
        njets = observations.shape[1]
        filter_width = observations.shape[2]
    if len(shape) == 3:
        njets = 1
        filter_width = observations.shape[1]
        observations = tf.reshape(observations, [-1, njets, filter_width, 2])

    # Layer sizes of the network that only belongs to the policy network
    policy_only_layers = []
    # Layer sizes of the network that only belongs to the value network
    value_only_layers = []

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            raise Exception(
                "You shouldn't be in there, have net_arch be a dict")
        else:
            assert isinstance(
                layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(
                    layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(
                    layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    latent = observations
    # set to true after we build the first mlp, so that all mlps are the same
    reuse_flag = False

    latent_policy = []
    latent_value = []

    for latent in tf.split(observations, njets, axis=1, name='splitting'):
        latent = tf.layers.flatten(latent)

        # Build the non-shared part of the network
        latent_policy_onejet = latent
        latent_value_onejet = latent

        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                latent_policy_onejet = act_fun(linear(latent_policy_onejet, "pi_fc{}".format(
                    idx), pi_layer_size, init_scale=np.sqrt(2), reuse=reuse_flag))

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                latent_value_onejet = act_fun(linear(latent_value_onejet, "vf_fc{}".format(
                    idx), vf_layer_size, init_scale=np.sqrt(2), reuse=reuse_flag))

        latent_policy.append(latent_policy_onejet)
        latent_value.append(latent_value_onejet)
        reuse_flag = True

    latent_policy = tf.concat(
        latent_policy,
        axis=1,
        name='concat_policy')
    print(latent_policy.shape)
    latent_value = tf.concat(
        latent_value,
        axis=1,
        name='concat_value')

    return latent_policy, latent_value


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
                                                      scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            assert feature_extraction != "cnn"
            pi_latent, vf_latent = mlp_extractor(
                self.processed_obs, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(
                    pi_latent, vf_latent, init_scale=0.01)

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


net_arch = param.arch


class CustomPolicy(CustomFeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=net_arch,
                                                          vf=net_arch)],
                                           feature_extraction="mlp")
