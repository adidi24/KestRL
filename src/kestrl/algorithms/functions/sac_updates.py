"""JIT-compiled update functions for SAC.

Pure functions that perform gradient computation + optimizer steps.
Each function takes all required nnx.Module objects as explicit arguments,
making them compatible with @nnx.jit without requiring the SAC class to
be an nnx.Module itself.

The pattern:
    @nnx.jit
    def update_critics(actor, critic1, critic2, ...):
        ...  # grad + optimizer step

    class SAC:
        def train_step(self):
            update_critics(self.actor, self.critic1, ...)
"""

import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.algorithms.functions.sac_losses import (
    compute_continuous_actor_loss,
    compute_continuous_critic_loss,
    compute_discrete_actor_loss,
    compute_discrete_critic_loss,
    get_continuous_actor_action,
    get_discrete_actor_action,
)

# ── Critic Discrete ────────────────────────────────────────────

@nnx.jit
def update_discrete(
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    target_critic1: nnx.Module,
    target_critic2: nnx.Module,
    actor_opt: nnx.Optimizer,
    critic1_opt: nnx.Optimizer,
    critic2_opt: nnx.Optimizer,
    log_alpha: nnx.Module,
    alpha_opt: nnx.Optimizer,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    dones: jax.Array,
    gamma: float,
    target_entropy: jax.Array,
    key: jax.Array,
) -> dict[str, jax.Array]:
    
    key1, key2 = jax.random.split(key)
    alpha = jnp.exp(log_alpha.value[...])
    
    # Compute target Q (no grad through actor/targets)
    _, next_log_pi, next_action_probs = get_discrete_actor_action(actor, next_obs, key1)
    min_next_q = jnp.minimum(
        target_critic1(next_obs), target_critic2(next_obs)
    ) - alpha * next_log_pi
    # Expectation over actions: sum(pi(a|s') * Q(s', a))
    min_next_q = jnp.sum(next_action_probs * min_next_q, axis=1)
    target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * min_next_q
    
    # Update critic1
    def c1_loss(c1):
        return compute_discrete_critic_loss(c1, target_q, obs, actions)
    q1_loss, grads1 = nnx.value_and_grad(c1_loss)(critic1)
    critic1_opt.update(critic1, grads1)

    # Update critic2
    def c2_loss(c2):
        return compute_discrete_critic_loss(c2, target_q, obs, actions)
    q2_loss, grads2 = nnx.value_and_grad(c2_loss)(critic2)
    critic2_opt.update(critic2, grads2)
    
    min_q_values = jnp.minimum(critic1(obs), critic2(obs))

    # Update actor via value_and_grad(has_aux=True) → get log_prob, action_probs
    def actor_loss_fn(actor):
        return compute_discrete_actor_loss(actor, min_q_values, obs, alpha, key2)

    (actor_loss, (log_prob, action_probs)), grads = nnx.value_and_grad(
        actor_loss_fn, has_aux=True)(actor)
    actor_opt.update(actor, grads)

    # Update alpha
    def alpha_loss_fn(la):
        return (action_probs * (
            -jnp.exp(la.value[...]) * (log_prob + target_entropy))).mean()

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_opt.update(log_alpha, alpha_grads)

    return {'critic/loss': q1_loss + q2_loss,
            'actor/loss': actor_loss,
            'alpha/loss': alpha_loss,
            'alpha/value': jnp.exp(log_alpha.value[...])}

# ── Critic Updates ────────────────────────────────────────────
@nnx.jit
def update_continuous_critics(
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    target_critic1: nnx.Module,
    target_critic2: nnx.Module,
    critic1_opt: nnx.Optimizer,
    critic2_opt: nnx.Optimizer,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    dones: jax.Array,
    log_alpha: nnx.Module,
    gamma: float,
    action_scale: jax.Array,
    action_bias: jax.Array,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Update twin critics for continuous action spaces."""
    alpha = jnp.exp(log_alpha.value[...])
    
    # Compute target Q
    next_actions, next_log_pi, _ = get_continuous_actor_action(
        actor, next_obs, action_scale, action_bias, key)
    tq_input = jnp.concatenate([next_obs, next_actions], axis=1)
    min_next_q = jnp.minimum(
        target_critic1(tq_input), target_critic2(tq_input)
    ) - alpha * next_log_pi
    target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * min_next_q.reshape(-1)

    # Critic 1
    def c1_loss(c1):
        return compute_continuous_critic_loss(c1, target_q, obs, actions)
    q1_loss, grads1 = nnx.value_and_grad(c1_loss)(critic1)
    critic1_opt.update(critic1, grads1)

    # Critic 2
    def c2_loss(c2):
        return compute_continuous_critic_loss(c2, target_q, obs, actions)
    q2_loss, grads2 = nnx.value_and_grad(c2_loss)(critic2)
    critic2_opt.update(critic2, grads2)

    return {'critic/loss': q1_loss + q2_loss}


# ── Actor Updates ─────────────────────────────────────────────

@nnx.jit(static_argnames='autotune_alpha')
def update_continuous_actor_alpha(
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    actor_opt: nnx.Optimizer,
    log_alpha: nnx.Module,
    alpha_opt: nnx.Optimizer,
    obs: jax.Array,
    action_scale: jax.Array,
    action_bias: jax.Array,
    autotune_alpha: bool,
    target_entropy: jax.Array,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Actor update for continuous action spaces."""
    key1, key2 = jax.random.split(key)
    alpha = jnp.exp(log_alpha.value[...])
    
    def loss_fn(actor):
        return compute_continuous_actor_loss(
            actor, critic1, critic2, obs, alpha, action_scale, action_bias, key1)

    actor_loss, grads = nnx.value_and_grad(loss_fn)(actor)
    actor_opt.update(actor, grads)
    
    if (autotune_alpha):
        _, log_probs, _ = get_continuous_actor_action(
            actor, obs, action_scale, action_bias, key2)

        def alpha_loss_fn(la):
            return (-jnp.exp(la.value[...]) * (log_probs + target_entropy)).mean()

        alpha_loss, grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)
        alpha_opt.update(log_alpha, grads)
        
        return {'actor/loss': actor_loss,
            'alpha/loss': alpha_loss,
            'alpha/value': jnp.exp(log_alpha.value[...])}
    
    return {'actor/loss': actor_loss}