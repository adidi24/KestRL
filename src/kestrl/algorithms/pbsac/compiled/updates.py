import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.algorithms.sac.functions import (
    compute_continuous_critic_loss,
    compute_discrete_critic_loss,
    get_continuous_actor_action,
    get_discrete_actor_action,
)
from kestrl.algorithms.pbsac.functions import (
    compute_pac_bayes_loss,
)
from kestrl.algorithms.pbsac.updates import (
    _make_frozen_inject_posterior,
)
from kestrl.distributions import (
    _construct_state_from_flat_state,
    block_sample,
    kl_block,
)

from kestrl.buffers import JAXBufferState, _buffer_sample

def _make_pb_posterior_scan(bundle_gd,
                        is_discrete: bool,
                        num_samples: int,
                        pb_update_epochs: int,
                        action_scale: jax.Array | None,
                        action_bias: jax.Array | None,
):
    """Return a scan-based posterior update function for PAC-Bayes SAC."""
    
    _inject_posterior = _make_frozen_inject_posterior(bundle_gd)
    
    def posterior_scan(bundle_state,
                       C_const,
                       C_prime_const,
                       traj_batch,
                       key
    ):
        batch = {
            'states':      traj_batch.states,
            'actions':     traj_batch.actions,
            'log_probs_b': traj_batch.log_probs_b,
            'masks':       traj_batch.masks,
            'returns':     traj_batch.returns,
        }
        if not is_discrete:
            batch['action_scale'] = action_scale
            batch['action_bias']  = action_bias

        def update_step(carry, _):
            carry_state, rng = carry
            rng, k_loss = jax.random.split(rng)

            b = nnx.merge(bundle_gd, carry_state)
            kl = kl_block(b.posterior, b.prior)
            lambda_val = jnp.sqrt(C_const * kl + C_prime_const)
            loss_fn = lambda p: compute_pac_bayes_loss(p, b.prior, b.actor,
                                                    is_discrete,
                                                    k_loss,
                                                    num_samples,
                                                    lambda_val,
                                                    C_const,
                                                    C_prime_const,
                                                    batch)
            (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(b.posterior)
            b.pb_optimizer.update(b.posterior, grads)

            _, new_state = nnx.split(b)
            return (new_state, rng), metrics
        
        (final_state, _), stacked_metrics = jax.lax.scan(
            update_step,
            (bundle_state, key),
            None,
            length=pb_update_epochs
        )
        
        final_state = _inject_posterior(final_state)
        
        avg_metrics = jax.tree.map(jnp.mean, stacked_metrics)
        return final_state, avg_metrics

    return posterior_scan

def _make_adaptive_gradient_scan_continuous(
    bundle_gd,
    batch_size: int,
    gradient_steps: int,
    action_scale,
    action_bias,
    gamma: float,
    adaptation_samples: int = 2,
):
    """Return a function that runs gradient_steps adaptive critic-only updates
    during the actor-freeze phase. Buffer sampling and gradient computations
    run inside a single lax.scan.
    """

    def gradient_scan(
        bundle_state,
        buf: JAXBufferState,
        key: jax.Array,
        upper: jax.Array,
    ):
        def update_step(carry, _):
            carry_state, rng = carry
            rng, k_sample, k_critic = jax.random.split(rng, 3)

            samples = _buffer_sample(buf, k_sample, batch_size, upper)
            obs = samples.observations
            next_obs = samples.next_observations
            actions = samples.actions
            rewards = samples.rewards
            dones = samples.dones

            b = nnx.merge(bundle_gd, carry_state)
            actor_gd, _ = nnx.split(b.actor)
            alpha = jnp.exp(b.log_alpha.value[...])

            def sample_and_apply(sample_key):
                sub_key1, sub_key2 = jax.random.split(sample_key)
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

            sa_keys = jax.random.split(k_critic, adaptation_samples)
            td_terms = jax.vmap(sample_and_apply)(sa_keys)
            target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * jnp.mean(td_terms, axis=0)

            def c1_loss(c1):
                return compute_continuous_critic_loss(c1, target_q, obs, actions)
            def c2_loss(c2):
                return compute_continuous_critic_loss(c2, target_q, obs, actions)
            
            q1_loss, g1 = nnx.value_and_grad(c1_loss)(b.critic1)
            q2_loss, g2 = nnx.value_and_grad(c2_loss)(b.critic2)
            b.critic1_opt.update(b.critic1, g1)
            b.critic2_opt.update(b.critic2, g2)

            metrics = {'critic/loss': q1_loss + q2_loss}

            _, new_state = nnx.split(b)
            return (new_state, rng), metrics
        
        (final_state, _), stacked_metrics = jax.lax.scan(
            update_step,
            (bundle_state, key),
            None,
            length=gradient_steps
        )
        
        avg_metrics = jax.tree.map(jnp.mean, stacked_metrics)
        return final_state, avg_metrics

    return gradient_scan

def _make_adaptive_gradient_scan_discrete(
    bundle_gd,
    batch_size: int,
    gradient_steps: int,
    gamma: float,
    adaptation_samples: int = 2,
):
    """Return a function that runs gradient_steps adaptive critic-only updates
    during the actor-freeze phase. Buffer sampling and gradient computations
    run inside a single lax.scan.
    """

    def gradient_scan(
        bundle_state,
        buf: JAXBufferState,
        key: jax.Array,
        upper: jax.Array,
    ):
        def update_step(carry, _):
            carry_state, rng = carry
            rng, k_sample, k_critic = jax.random.split(rng, 3)

            samples = _buffer_sample(buf, k_sample, batch_size, upper)
            obs = samples.observations
            next_obs = samples.next_observations
            actions = samples.actions
            rewards = samples.rewards
            dones = samples.dones

            b = nnx.merge(bundle_gd, carry_state)
            actor_gd, _ = nnx.split(b.actor)
            alpha = jnp.exp(b.log_alpha.value[...])
            
            # ── Critics ───────────────────────────────────────
            # Sample actor weights from posterior → action distribution → soft next-Q
            def sample_and_apply(sample_key):
                sub_key1, sub_key2 = jax.random.split(sample_key)
                flat_state = block_sample(b.posterior, sub_key1)
                sampled_actor_state = _construct_state_from_flat_state(flat_state, b.posterior.shapes)
                temp_actor = nnx.merge(actor_gd, sampled_actor_state)
                _, next_log_pi, next_action_probs = get_discrete_actor_action(temp_actor, next_obs, sub_key2)
                min_next_q = jnp.minimum(
                    b.target_critic1(next_obs), b.target_critic2(next_obs)
                ) - alpha * next_log_pi
                min_next_q = jnp.sum(next_action_probs * min_next_q, axis=1)
                return min_next_q

            sa_keys = jax.random.split(k_critic, adaptation_samples)
            td_terms = jax.vmap(sample_and_apply)(sa_keys)
            target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * jnp.mean(td_terms, axis=0)

            def c1_loss(c1):
                return compute_discrete_critic_loss(c1, target_q, obs, actions)
            def c2_loss(c2):
                return compute_discrete_critic_loss(c2, target_q, obs, actions)
            
            q1_loss, g1 = nnx.value_and_grad(c1_loss)(b.critic1)
            q2_loss, g2 = nnx.value_and_grad(c2_loss)(b.critic2)
            b.critic1_opt.update(b.critic1, g1)
            b.critic2_opt.update(b.critic2, g2)

            metrics = {'critic/loss': q1_loss + q2_loss}

            _, new_state = nnx.split(b)
            return (new_state, rng), metrics
        
        (final_state, _), stacked_metrics = jax.lax.scan(
            update_step,
            (bundle_state, key),
            None,
            length=gradient_steps
        )
        
        avg_metrics = jax.tree.map(jnp.mean, stacked_metrics)
        return final_state, avg_metrics

    return gradient_scan