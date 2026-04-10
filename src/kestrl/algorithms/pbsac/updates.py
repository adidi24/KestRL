"""Frozen pytree update functions for PB-SAC.

graphdef is captured in Python closure at init; only state pytrees cross the
JIT boundary. Factories return plain callables — callers apply jax.jit so the
same function can be used unjitted inside lax.scan bodies.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from kestrl.algorithms.sac.functions import (
    compute_discrete_critic_loss,
    compute_continuous_critic_loss,
    get_discrete_actor_action,
    get_continuous_actor_action,
)
from kestrl.algorithms.pbsac.functions import (
    compute_pac_bayes_loss,
    compute_pac_bayes_bound,
    get_actor_discrete_action_from_posterior_vmap,
    get_actor_continuous_action_from_posterior_vmap,
)
from kestrl.distributions import (
    _construct_state_from_flat_state,
    block_sample,
    ema_update_prior,
    kl_block,
)

def _make_frozen_update_pb_posterior(bundle_gd, is_discrete: bool, num_samples: int):
    
    def _update(
        bundle_state,
        C_const, C_prime_const,
        batch_data,
        key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        kl = kl_block(b.posterior, b.prior)
        lambda_val = jnp.sqrt(C_const * kl + C_prime_const)
        loss_fn = lambda p: compute_pac_bayes_loss(p, b.prior, b.actor,
                                                is_discrete,
                                                key,
                                                num_samples,
                                                lambda_val,
                                                C_const,
                                                C_prime_const,
                                                batch_data)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(b.posterior)
        b.pb_optimizer.update(b.posterior, grads)
        _, new_state = nnx.split(b)
        return new_state, metrics
    
    def _compute_bound(
        bundle_state,
        T: int,
        H: int,
        r_max: float,
        mixing_time: int,
        gamma: float,
        delta: float,
        batch_data: dict,
        key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        return compute_pac_bayes_bound(b.posterior, b.prior, b.actor,
                                        is_discrete,
                                        key,
                                        num_samples,
                                        T,
                                        H,
                                        r_max,
                                        mixing_time,
                                        gamma,
                                        delta,
                                        batch_data)
    
    return _update, _compute_bound

# ── Critic Discrete Update ────────────────────────────────────────────

def _make_frozen_adaptive_discrete_update(bundle_gd, adaptation_samples: int = 2):
    """Discrete PB-SAC critic update during the actor-frozen phase.

    Averages TD targets over `adaptation_samples` actor weight draws from the
    posterior, then updates critics only (actor stays frozen).
    `adaptation_samples` is captured in the closure so jax.random.split gets a
    static int — not a traced value.
    """

    def _update(
        bundle_state,
        obs, actions, rewards, next_obs, dones,
        gamma, key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        alpha = jnp.exp(b.log_alpha.value[...])
        # Graphdef extracted once outside vmap; only key is mapped, not the module structure
        actor_gd, _ = nnx.split(b.actor)

        # Sample actor weights from posterior → action distribution → soft next-Q
        def sample_and_apply(key):
            sub_key1, sub_key2 = jax.random.split(key)
            flat_state = block_sample(b.posterior, sub_key1)
            sampled_actor_state = _construct_state_from_flat_state(flat_state, b.posterior.shapes)
            temp_actor = nnx.merge(actor_gd, sampled_actor_state)
            _, next_log_pi, next_action_probs = get_discrete_actor_action(temp_actor, next_obs, sub_key2)
            min_next_q = jnp.minimum(
                b.target_critic1(next_obs), b.target_critic2(next_obs)
            ) - alpha * next_log_pi
            min_next_q = jnp.sum(next_action_probs * min_next_q, axis=1)
            return min_next_q

        sa_keys = jax.random.split(key, adaptation_samples)
        td_terms = jax.vmap(sample_and_apply)(sa_keys)
        target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * jnp.mean(td_terms, axis=0)

        def c1_loss(c1):
            return compute_discrete_critic_loss(c1, target_q, obs, actions)
        q1_loss, grads1 = nnx.value_and_grad(c1_loss)(b.critic1)
        b.critic1_opt.update(b.critic1, grads1)

        # Update critic2
        def c2_loss(c2):
            return compute_discrete_critic_loss(c2, target_q, obs, actions)
        q2_loss, grads2 = nnx.value_and_grad(c2_loss)(b.critic2)
        b.critic2_opt.update(b.critic2, grads2)

        metrics = {'critic/loss': q1_loss + q2_loss}

        _, new_state = nnx.split(b)
        return new_state, metrics

    return _update

# ── Critic Continuous Updates ────────────────────────────────────────────
def _make_frozen_adaptive_continuous_update(bundle_gd, action_scale, action_bias, adaptation_samples: int = 2):
    """Continuous PB-SAC critic update during the actor-frozen phase.

    Same structure as the discrete version. Critics must be inside vmap here
    because their input (next_action) depends on the sampled actor weights.
    """

    def _update(
        bundle_state,
        obs, actions, rewards, next_obs, dones,
        gamma, key,
    ):
        b = nnx.merge(bundle_gd, bundle_state)
        alpha = jnp.exp(b.log_alpha.value[...])
        actor_gd, _ = nnx.split(b.actor)

        # Sample actor weights from posterior → sampled next_action → soft next-Q
        def sample_and_apply(key):
            sub_key1, sub_key2 = jax.random.split(key)
            flat_state = block_sample(b.posterior, sub_key1)
            sampled_actor_state = _construct_state_from_flat_state(flat_state, b.posterior.shapes)
            temp_actor = nnx.merge(actor_gd, sampled_actor_state)
            next_action, next_log_pi, _ = get_continuous_actor_action(
                temp_actor, next_obs, action_scale, action_bias, sub_key2
            )
            critics_input = jnp.concatenate([next_obs, next_action], axis=1)
            min_next_q = jnp.minimum(
                b.target_critic1(critics_input),
                b.target_critic2(critics_input),
            ) - alpha * next_log_pi
            return min_next_q.reshape(-1)

        sa_keys = jax.random.split(key, adaptation_samples)
        td_terms = jax.vmap(sample_and_apply)(sa_keys)
        target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * jnp.mean(td_terms, axis=0)

        # Critic 1
        def c1_loss(c1):
            return compute_continuous_critic_loss(c1, target_q, obs, actions)
        q1_loss, grads1 = nnx.value_and_grad(c1_loss)(b.critic1)
        b.critic1_opt.update(b.critic1, grads1)

        # Critic 2
        def c2_loss(c2):
            return compute_continuous_critic_loss(c2, target_q, obs, actions)
        q2_loss, grads2 = nnx.value_and_grad(c2_loss)(b.critic2)
        b.critic2_opt.update(b.critic2, grads2)

        metrics = {'critic/loss': q1_loss + q2_loss}

        _, new_state = nnx.split(b)
        return new_state, metrics

    return _update

def _make_frozen_sync_posterior(bundle_gd):
    """Copy current actor params → posterior layer means after each SAC update."""

    def _sync(bundle_state):
        b = nnx.merge(bundle_gd, bundle_state)
        params_tree = nnx.state(b.actor, nnx.Param)
        for path, param in nnx.to_flat_state(params_tree):
            b.posterior.layers[path].mean.set_value(param.get_value().flatten())
        _, new_state = nnx.split(b)
        return new_state

    return _sync


def _make_frozen_inject_posterior(bundle_gd):
    """Overwrite actor params with posterior mean (no noise) after PAC-Bayes update."""

    def _inject(bundle_state):
        b = nnx.merge(bundle_gd, bundle_state)
        flat_state = {name: lp.mean.get_value() for name, lp in b.posterior.layers.items()}
        actor_state = _construct_state_from_flat_state(flat_state, b.posterior.shapes)
        nnx.update(b.actor, actor_state)
        _, new_state = nnx.split(b)
        return new_state

    return _inject


def _make_frozen_prior_ema_update(bundle_gd, decay: float):
    """Slide prior toward posterior: μ₀ ← decay·μ_q + (1-decay)·μ₀."""

    def _ema_update(bundle_state):
        b = nnx.merge(bundle_gd, bundle_state)
        ema_update_prior(b.prior, b.posterior, decay)
        _, new_state = nnx.split(b)
        return new_state

    return _ema_update


def _make_frozen_discrete_get_action_pb(bundle_gd, explore_n_samples: int):
    
    def _sample(bundle_state, obs, should_explore, key):
        b = nnx.merge(bundle_gd, bundle_state)
        return get_actor_discrete_action_from_posterior_vmap(b.posterior, b.actor, b.critic1, b.critic2,
                                                             obs, should_explore, explore_n_samples, key)

    def _deterministic(bundle_state, obs):
        b = nnx.merge(bundle_gd, bundle_state)
        return jnp.argmax(b.actor(obs), axis=-1), None, None

    return _sample, _deterministic

def _make_frozen_continuous_get_action_pb(bundle_gd, action_scale, action_bias, explore_n_samples: int):
    
    def _sample(bundle_state, obs, should_explore, key):
        b = nnx.merge(bundle_gd, bundle_state)
        return get_actor_continuous_action_from_posterior_vmap(b.posterior, b.actor, b.critic1, b.critic2,
                                                               obs, action_scale, action_bias, should_explore, explore_n_samples, key)

    def _deterministic(bundle_state, obs, action_scale, action_bias):
        b = nnx.merge(bundle_gd, bundle_state)
        logits = b.actor(obs)
        return jnp.tanh(logits['mean']) * action_scale + action_bias, None, None

    return _sample, _deterministic