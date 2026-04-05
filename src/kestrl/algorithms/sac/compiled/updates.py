"""Gradient-step scan for CompiledSAC — eliminates the Python loop over gradient_steps.

Each factory returns a single @jax.jit function:

    gradient_scan(bundle_state, buf, key, upper) → (bundle_state, avg_metrics)

Where:
    bundle_state : SAC state pytree — carry in and out
    buf          : JAXBufferState — explicit JIT arg so updated arrays are traced
    key          : PRNGKey — split internally to generate per-step sub-keys
    upper        : int32 JAX scalar — valid row upper bound in buf

The lax.scan body runs one full SAC update per step (critics + actor + alpha).
Metrics are averaged across gradient_steps before returning.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.algorithms.sac.functions import (
    compute_continuous_critic_loss,
    compute_continuous_actor_loss,
    compute_discrete_critic_loss,
    compute_discrete_actor_loss,
    get_continuous_actor_action,
    get_discrete_actor_action,
)
from kestrl.buffers import JAXBufferState, _buffer_sample

def _make_gradient_scan_continuous(
    bundle_gd,
    batch_size: int,
    gradient_steps: int,
    action_scale,
    action_bias,
    target_entropy,
    gamma: float,
    autotune_alpha: bool,
):
    """Return a JIT'd function that runs gradient_steps full continuous SAC updates.

    Buffer sampling and all gradient computations happen inside a single lax.scan,
    so the entire block compiles to one XLA dispatch.
    """
    
    def gradient_scan(
        bundle_state,
        buf: JAXBufferState,
        key: jax.Array,
        upper: jax.Array,
    ):
        def update_step(carry, _):
            carry_state, rng = carry
            rng, k_sample, k_critic, k_actor = jax.random.split(rng, 4)
            
            samples = _buffer_sample(buf, k_sample, batch_size, upper)
            obs = samples.observations
            next_obs = samples.next_observations
            actions = samples.actions
            rewards = samples.rewards
            dones = samples.dones
            
            b = nnx.merge(bundle_gd, carry_state)
            alpha = jnp.exp(b.log_alpha.value[...])
            
            # ── Critics ───────────────────────────────────────
            next_act, next_log_pi, _ = get_continuous_actor_action(
                b.actor, next_obs, action_scale, action_bias, k_critic
            )
            tq_input = jnp.concatenate([next_obs, next_act], axis=1)
            min_next_q = (
                jnp.minimum(b.target_critic1(tq_input), b.target_critic2(tq_input))
                - alpha * next_log_pi
            )
            target_q = (
                rewards.flatten()
                + (1 - dones.flatten()) * gamma * min_next_q.reshape(-1)
            )
            
            def c1_loss(c1):
                return compute_continuous_critic_loss(c1, target_q, obs, actions)
            def c2_loss(c2):
                return compute_continuous_critic_loss(c2, target_q, obs, actions)
            
            q1_loss, g1 = nnx.value_and_grad(c1_loss)(b.critic1)
            q2_loss, g2 = nnx.value_and_grad(c2_loss)(b.critic2)
            b.critic1_opt.update(b.critic1, g1)
            b.critic2_opt.update(b.critic2, g2)
            
            # ── Actor ─────────────────────────────────────────
            ka1, ka2 = jax.random.split(k_actor)
            
            def actor_loss_fn(actor):
                return compute_continuous_actor_loss(
                    actor, b.critic1, b.critic2, obs, alpha,
                    action_scale, action_bias, ka1,
                )

            actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(b.actor)
            b.actor_opt.update(b.actor, actor_grads)

            metrics = {'critic/loss': q1_loss + q2_loss, 'actor/loss': actor_loss}
            
            # ── Alpha ─────────────────────────────────────────
            if autotune_alpha:
                _, log_probs, _ = get_continuous_actor_action(
                    b.actor, obs, action_scale, action_bias, ka2
                )

                def alpha_loss_fn(la):
                    return (-jnp.exp(la.value[...]) * (log_probs + target_entropy)).mean()

                alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(b.log_alpha)
                b.alpha_opt.update(b.log_alpha, alpha_grads)
                metrics['alpha/loss']  = alpha_loss
                metrics['alpha/value'] = jnp.exp(b.log_alpha.value[...])

            _, new_state = nnx.split(b)
            return (new_state, rng), metrics
        
        (final_state, _), stacked_metrics = jax.lax.scan(
            update_step, (bundle_state, key), None, length=gradient_steps
        )
        avg_metrics = jax.tree.map(jnp.mean, stacked_metrics)
        return final_state, avg_metrics

    return gradient_scan


def _make_gradient_scan_discrete(
    bundle_gd,
    batch_size: int,
    gradient_steps: int,
    target_entropy,
    gamma: float,
    autotune_alpha: bool,
):
    """Return a JIT'd function that runs gradient_steps full discrete SAC updates."""
    def gradient_scan(
        bundle_state,
        buf: JAXBufferState,
        key: jax.Array,
        upper: jax.Array,
    ):
        def update_step(carry, _):
            carry_state, rng = carry
            rng, k_sample, k_update = jax.random.split(rng, 3)
            
            samples = _buffer_sample(buf, k_sample, batch_size, upper)
            obs = samples.observations
            next_obs = samples.next_observations
            actions = samples.actions
            rewards = samples.rewards
            dones = samples.dones
            
            b = nnx.merge(bundle_gd, carry_state)
            alpha = jnp.exp(b.log_alpha.value[...])
            
            # ── Full discrete SAC update ──────────────────────
            ku1, ku2 = jax.random.split(k_update)

            next_action, next_log_pi, next_action_probs = get_discrete_actor_action(
                b.actor, next_obs, ku1
            )
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
                return compute_discrete_actor_loss(actor, min_q_values, obs, alpha, ku2)

            (actor_loss, (log_prob, action_probs)), actor_grads = nnx.value_and_grad(
                actor_loss_fn, has_aux=True
            )(b.actor)
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
            return (new_state, rng), metrics
        
        (final_state, _), stacked_metrics = jax.lax.scan(
            update_step, (bundle_state, key), None, length=gradient_steps
        )
        avg_metrics = jax.tree.map(jnp.mean, stacked_metrics)
        return final_state, avg_metrics

    return gradient_scan