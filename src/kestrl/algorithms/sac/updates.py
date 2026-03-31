"""Frozen pytree-first SAC update functions.

graphdef is captured in Python closure at init; only state pytrees flow through
@jax.jit. nnx.merge/split inside JIT are trace-time ops, no Python overhead
per call vs @nnx.jit which re-traverses the object graph every call.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from distrax import Categorical, Normal

from kestrl.algorithms.sac.functions import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    compute_continuous_critic_loss,
    compute_discrete_critic_loss,
    compute_continuous_actor_loss,
    compute_discrete_actor_loss,
    get_continuous_actor_action,
    get_discrete_actor_action,
)


def _make_frozen_continuous_update(bundle_gd, autotune_alpha: bool):
    """Return (update_critics_fn, update_actor_alpha_fn)."""

    @jax.jit
    def _update_critics(
        bundle_state,
        obs, actions, rewards, next_obs, dones,
        gamma, action_scale, action_bias, key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        alpha = jnp.exp(b.log_alpha.value[...])

        next_act, next_log_pi, _ = get_continuous_actor_action(b.actor, next_obs, action_scale, action_bias, key)
        tq_input = jnp.concatenate([next_obs, next_act], axis=1)
        min_next_q = (
            jnp.minimum(b.target_critic1(tq_input), b.target_critic2(tq_input))
            - alpha * next_log_pi
        )
        target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * min_next_q.reshape(-1)

        def c1_loss(c1):
            return compute_continuous_critic_loss(c1, target_q, obs, actions)
        def c2_loss(c2):
            return compute_continuous_critic_loss(c2, target_q, obs, actions)
        q1_loss, g1 = nnx.value_and_grad(c1_loss)(b.critic1)
        q2_loss, g2 = nnx.value_and_grad(c2_loss)(b.critic2)
        b.critic1_opt.update(b.critic1, g1)
        b.critic2_opt.update(b.critic2, g2)

        _, new_state = nnx.split(b)
        return new_state, {'critic/loss': q1_loss + q2_loss}

    @jax.jit
    def _update_actor_alpha(
        bundle_state,
        obs, action_scale, action_bias, target_entropy, key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        key1, key2 = jax.random.split(key)
        alpha = jnp.exp(b.log_alpha.value[...])

        def actor_loss_fn(actor):
            return compute_continuous_actor_loss(actor, b.critic1, b.critic2, obs, alpha, action_scale, action_bias, key1)
        actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(b.actor)
        b.actor_opt.update(b.actor, actor_grads)

        metrics = {'actor/loss': actor_loss}

        if autotune_alpha:
            _, log_probs, _ = get_continuous_actor_action(b.actor, obs, action_scale, action_bias, key2)
            def alpha_loss_fn(la):
                return (-jnp.exp(la.value[...]) * (log_probs + target_entropy)).mean()
            alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(b.log_alpha)
            b.alpha_opt.update(b.log_alpha, alpha_grads)
            metrics['alpha/loss']  = alpha_loss
            metrics['alpha/value'] = jnp.exp(b.log_alpha.value[...])

        _, new_state = nnx.split(b)
        return new_state, metrics

    return _update_critics, _update_actor_alpha


def _make_frozen_soft_update(bundle_gd):
    @jax.jit
    def _soft_update(bundle_state, tau):
        b = nnx.merge(bundle_gd, bundle_state)
        c1_state = nnx.state(b.critic1)
        c2_state = nnx.state(b.critic2)
        t1_state = nnx.state(b.target_critic1)
        t2_state = nnx.state(b.target_critic2)
        new_t1 = jax.tree.map(lambda t, o: (1 - tau) * t + tau * o, t1_state, c1_state)
        new_t2 = jax.tree.map(lambda t, o: (1 - tau) * t + tau * o, t2_state, c2_state)
        nnx.update(b.target_critic1, new_t1)
        nnx.update(b.target_critic2, new_t2)
        _, new_state = nnx.split(b)
        return new_state

    return _soft_update


def _make_frozen_continuous_get_action(bundle_gd):
    @jax.jit
    def _sample(bundle_state, obs, action_scale, action_bias, key):
        b = nnx.merge(bundle_gd, bundle_state)
        return get_continuous_actor_action(b.actor, obs, action_scale, action_bias, key)

    @jax.jit
    def _deterministic(bundle_state, obs, action_scale, action_bias):
        b = nnx.merge(bundle_gd, bundle_state)
        logits = b.actor(obs)
        return jnp.tanh(logits['mean']) * action_scale + action_bias, None, None

    return _sample, _deterministic


def _make_frozen_discrete_update(bundle_gd, autotune_alpha: bool):
    """Discrete SAC update. No sampling needed — all expectations are over the
    full action distribution via softmax/log_softmax."""

    @jax.jit
    def _update(
        bundle_state,
        obs, actions, rewards, next_obs, dones,
        gamma, target_entropy, key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        alpha = jnp.exp(b.log_alpha.value[...])
        key1, key2 = jax.random.split(key)
        
        next_action, next_log_pi, next_action_probs = get_discrete_actor_action(b.actor, next_obs, key1)
        min_next_q = (
            jnp.minimum(b.target_critic1(next_obs), b.target_critic2(next_obs))
            - alpha * next_log_pi
        )
        min_next_q = jnp.sum(next_action_probs * min_next_q, axis=1)
        target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * min_next_q

        def c1_loss(c1):
            return compute_discrete_critic_loss(c1, target_q, obs, actions)
        def c2_loss(c2):
            return compute_discrete_critic_loss(c2, target_q, obs, actions)

        q1_loss, g1 = nnx.value_and_grad(c1_loss)(b.critic1)
        q2_loss, g2 = nnx.value_and_grad(c2_loss)(b.critic2)
        b.critic1_opt.update(b.critic1, g1)
        b.critic2_opt.update(b.critic2, g2)

        min_q_values = jnp.minimum(b.critic1(obs), b.critic2(obs))

        def actor_loss_fn(actor):
            return compute_discrete_actor_loss(actor, min_q_values, obs, alpha, key2)

        (actor_loss, (log_prob, action_probs)), actor_grads = nnx.value_and_grad(
            actor_loss_fn, has_aux=True)(b.actor)
        b.actor_opt.update(b.actor, actor_grads)

        metrics = {'critic/loss': q1_loss + q2_loss, 'actor/loss': actor_loss}

        if autotune_alpha:
            def alpha_loss_fn(la):
                return (action_probs * (
                    -jnp.exp(la.value[...]) * (log_prob + target_entropy)
                )).mean()
            alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(b.log_alpha)
            b.alpha_opt.update(b.log_alpha, alpha_grads)
            metrics['alpha/loss']  = alpha_loss
            metrics['alpha/value'] = jnp.exp(b.log_alpha.value[...])

        _, new_state = nnx.split(b)
        return new_state, metrics

    return _update


def _make_frozen_discrete_get_action(bundle_gd):
    @jax.jit
    def _sample(bundle_state, obs, key):
        b = nnx.merge(bundle_gd, bundle_state)
        return get_discrete_actor_action(b.actor, obs, key)

    @jax.jit
    def _deterministic(bundle_state, obs):
        b = nnx.merge(bundle_gd, bundle_state)
        return jnp.argmax(b.actor(obs), axis=-1), None, None

    return _sample, _deterministic
