from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Basic example of a DQN policy without any optimizations."""

from gym.spaces import Discrete, Continuous
import logging

import ray
from ray.rllib.agents.dqn.simple_q_model import SimpleQModel
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.tf_ops import huber_loss

tf = try_import_tf()
logger = logging.getLogger(__name__)

Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"


class ExplorationStateMixin(object):
    def __init__(self, obs_space, action_space, config):
        self.cur_epsilon = 1.0
        self.stochastic = tf.placeholder(tf.bool, (), name="stochastic")
        self.eps = tf.placeholder(tf.float32, (), name="eps")

    def add_parameter_noise(self):
        if self.config["parameter_noise"]:
            self.sess.run(self.add_noise_op)

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    @override(Policy)
    def get_state(self):
        return [TFPolicy.get_state(self), self.cur_epsilon]

    @override(Policy)
    def set_state(self, state):
        TFPolicy.set_state(self, state[0])
        self.set_epsilon(state[1])


class TargetNetworkMixin(object):
    def __init__(self, obs_space, action_space, config):
        # update_target_fn will be called periodically to copy Q network to
        # target Q network
        update_target_expr = []
        assert len(self.q_func_vars) == len(self.target_q_func_vars), \
            (self.q_func_vars, self.target_q_func_vars)
        for var, var_target in zip(self.q_func_vars, self.target_q_func_vars):
            update_target_expr.append(var_target.assign(var))
            logger.debug("Update target op {}".format(var_target))
        self.update_target_expr = tf.group(*update_target_expr)

    def update_target(self):
        return self.get_session().run(self.update_target_expr)


def build_q_models(policy, obs_space, action_space, config):

    if not isinstance(action_space, Continuous):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    if config["hiddens"]:
        num_outputs = 256
        config["model"]["no_final_linear"] = True
    else:
        num_outputs = action_space.n

    policy.q_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        num_outputs,
        config["model"],
        framework="tf",
        name=Q_SCOPE,
        model_interface=SimpleQModel,
        q_hiddens=config["hiddens"])

    policy.target_q_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        num_outputs,
        config["model"],
        framework="tf",
        name=Q_TARGET_SCOPE,
        model_interface=SimpleQModel,
        q_hiddens=config["hiddens"])

    return policy.q_model


def build_action_sampler(policy, q_model, input_dict, obs_space, action_space,
                         config):

    # Action Q network
    q_values = _compute_q_values(policy, q_model,
                                 input_dict[SampleBatch.CUR_OBS], obs_space,
                                 action_space)
    policy.q_values = q_values
    policy.q_func_vars = q_model.variables()

    # Action outputs
    deterministic_actions = tf.argmax(q_values, axis=1)
    batch_size = tf.shape(input_dict[SampleBatch.CUR_OBS])[0]

    # Special case masked out actions (q_value ~= -inf) so that we don't
    # even consider them for exploration.
    random_valid_action_logits = tf.where(
        tf.equal(q_values, tf.float32.min),
        tf.ones_like(q_values) * tf.float32.min, tf.ones_like(q_values))
    random_actions = tf.squeeze(
        tf.multinomial(random_valid_action_logits, 1), axis=1)

    chose_random = tf.random_uniform(
        tf.stack([batch_size]), minval=0, maxval=1,
        dtype=tf.float32) < policy.eps
    stochastic_actions = tf.where(chose_random, random_actions,
                                  deterministic_actions)
    action = tf.cond(policy.stochastic, lambda: stochastic_actions,
                     lambda: deterministic_actions)
    action_logp = None

    return action, action_logp


def build_q_losses(policy, batch_tensors):
    # q network evaluation
    q_t = _compute_q_values(policy, policy.q_model,
                            batch_tensors[SampleBatch.CUR_OBS],
                            batch_tensors[SampleBatch.ACTIONS],
                            policy.observation_space, policy.action_space)

    # target q network evalution
    q_tp1 = _compute_q_values(policy, policy.target_q_model,
                              batch_tensors[SampleBatch.NEXT_OBS],
                              None,
                              policy.observation_space, policy.action_space)
    policy.target_q_func_vars = policy.target_q_model.variables()


    # compute estimate of best possible value starting from state at t + 1
    dones = tf.cast(batch_tensors[SampleBatch.DONES], tf.float32)

    q_tp1_masked = (1.0 - dones) * q_tp1

    # compute RHS of bellman equation
    q_t_selected_target = (batch_tensors[SampleBatch.REWARDS] +
                           policy.config["gamma"] * q_tp1_masked)

    # compute the error (potentially clipped)
    td_error = q_t - tf.stop_gradient(q_t_selected_target)
    loss = tf.reduce_mean(huber_loss(td_error))

    # save TD error as an attribute for outside access
    policy.td_error = td_error

    return loss


def _compute_q_values(policy, model, obs, actions, obs_space, action_space):
    input_dict = {
        "obs": obs,
        "is_training": policy._get_is_training_placeholder(),
    }
    model_out, _ = model(input_dict, [], None)
    return model.get_q_values(model_out, actions)


def exploration_setting_inputs(policy):
    return {
        policy.stochastic: True,
        policy.eps: policy.cur_epsilon,
    }


def setup_early_mixins(policy, obs_space, action_space, config):
    ExplorationStateMixin.__init__(policy, obs_space, action_space, config)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


SimpleQPolicy = build_tf_policy(
    name="SimpleQPolicy",
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model=build_q_models,
    action_sampler_fn=build_action_sampler,
    loss_fn=build_q_losses,
    extra_action_feed_fn=exploration_setting_inputs,
    extra_action_fetches_fn=lambda policy: {"q_values": policy.q_values},
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    before_init=setup_early_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        ExplorationStateMixin,
        TargetNetworkMixin,
    ])
