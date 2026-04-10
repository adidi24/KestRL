"""Implementation of Pac-Bayes rollout collection using JAX scan.
Collected trajectories are used to optimise and evaluate the PAC-Bayes bound.

SAC rollout collection is implemented in kestrl.algorithms.sac.compiled.rollout.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.algorithms.sac.functions import (
    get_continuous_actor_action,
    get_discrete_actor_action,
)
from kestrl.buffers import JAXTransition

from kestrl.algorithms.pbsac.functions import (
    get_actor_discrete_action_from_posterior_vmap,
    get_actor_continuous_action_from_posterior_vmap,
)

def _make_jax_scan_pge_rollout(
    bundle_gd,
    env,
    is_discrete: bool,
    action_scale: jax.Array | None,
    action_bias: jax.Array | None,
    explore_n_samples: int,
):
    """Return a scan-based rollout function for a Brax/MJX environment.

    Args:
        bundle_gd    : SAC bundle graphdef, captured in closure. Never re-traced.
        env          : Batched Brax/MJX env. Its .step() is JAX-traceable and safe
                       to call inside lax.scan.
        is_discrete  : Resolved at factory time (no Python branch inside JIT).
        action_scale : jax.Array for continuous action rescaling, or None for discrete.
        action_bias  : jax.Array for continuous action shifting, or None for discrete.

    Returns:
        collect_rollout: JIT'd function with signature:
          (bundle_state, jax_env_state, prng_keys) → (final_jax_env_state, JAXTransition)
    """
    if is_discrete:
        def _act(bundle_state, obs, should_explore, key):
            action, _, _ = get_actor_discrete_action_from_posterior_vmap(bundle_state.posterior, bundle_state.actor, bundle_state.critic1, bundle_state.critic2,
                                                                obs, should_explore, explore_n_samples, key)
            return action
    else:
        assert action_scale is not None and action_bias is not None, "action_scale and action_bias must be provided for continuous environments"
        def _act(bundle_state, obs, should_explore, key):
            action, _, _ = get_actor_continuous_action_from_posterior_vmap(
                bundle_state.posterior, bundle_state.actor, bundle_state.critic1, bundle_state.critic2,
                obs, action_scale, action_bias, should_explore, explore_n_samples, key
            )
            return action
    
    def collect_rollout(bundle_state, jax_env_state, prng_keys, should_explore):
        """Collect num_steps transitions via lax.scan.

        Args:
            bundle_state: SAC state pytree. Actor weights are read; nothing is mutated.
            jax_env_state  : Starting Brax/MJX State pytree. Used as initial scan carry.
            prng_keys   : (num_steps, 2) array of PRNGKeys, one consumed per step.

        Returns:
            (final_jax_env_state, trajectory)
              final_jax_env_state: Brax/MJX State after all steps, ready for next call.
              trajectory      : JAXTransition with leaves (num_steps, num_envs, ...).
        """
        # Merge at trace time
        b = nnx.merge(bundle_gd, bundle_state)
        
        def step_fn(env_state, key):
            obs = env_state.obs
            action = _act(b, obs, should_explore, key)
            next_env_state = env.step(env_state, action)
            transition = JAXTransition(
                observations = obs,
                actions = action,
                next_observations = next_env_state.obs,
                rewards = next_env_state.reward,
                dones = next_env_state.done,
            )
            return next_env_state, transition
        
        final_jax_env_state, trajectory  = jax.lax.scan(
            step_fn, jax_env_state, prng_keys
        )
        return final_jax_env_state, trajectory
    
    return collect_rollout
    

class PBTrajectoryBatch(NamedTuple):
    """ A batch of trajectories of experience.
    Each field is a batch of trajectories of shape `(num_traj, pb_rollout_steps, ...)`.
    """
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    log_probs_b: jax.Array
    masks: jax.Array
    returns: jax.Array

def _make_jax_pb_rollout(
    bundle_gd,
    pb_env,                # raw brax_env (pb_env.brax_env)
    num_envs: int,
    pb_rollout_trajectories: int,
    pb_rollout_steps: int,
    gamma: float,
    is_discrete: bool,
    action_scale: jax.Array | None,
    action_bias: jax.Array | None,
):
    """Return a scan-based rollout function for PAC-Bayes trajectory collection.

    Args:
        bundle_gd    : SAC/PB-SAC bundle graphdef, captured in closure.
        pb_env       : Raw Brax pipeline env (pb_env.brax_env). Its .reset(key)
                       and .step(state, action) are JAX-native and safe inside lax.scan.
        num_envs     : Number of parallel envs (pb_env.num_envs from the wrapper).
        is_discrete  : Resolved at factory time — no Python branch inside JIT.
        action_scale : jax.Array for continuous rescaling, or None for discrete.
        action_bias  : jax.Array for continuous shifting, or None for discrete.

    Returns:
        collect_fn: (bundle_state, prng_key) → (PBTrajectoryBatch, r_max)
    """
    if is_discrete:
        def _act(actor, obs, key):
            action, log_probs_b, _ = get_discrete_actor_action(actor, obs, key)
            return action, log_probs_b
    else:
        assert action_scale is not None and action_bias is not None, \
            "action_scale and action_bias must be provided for continuous environments"
        def _act(actor, obs, key):
            action, log_probs_b, _ = get_continuous_actor_action(
                actor, obs, action_scale, action_bias, key
            )
            return action, log_probs_b

    batches_needed = (pb_rollout_trajectories + num_envs - 1) // num_envs

    def collect_fn(bundle_state, prng_key):
        b = nnx.merge(bundle_gd, bundle_state)
        actor = b.actor
        
        def batch_fn(r_max, key):
            k_reset, k_steps = jax.random.split(key)
            env_state = pb_env.reset(k_reset)
            
            def step_fn(batch_carry, batch_key):
                env_state, r_max, active = batch_carry
                obs = env_state.obs
                action, log_probs_b = _act(actor, obs, batch_key)
                if is_discrete:
                    # get_discrete_actor_action returns log_softmax over all actions;
                    # IS ratio needs only the log prob of the action taken
                    log_probs_b = log_probs_b[jnp.arange(num_envs), action.astype(int)]
                else:
                    log_probs_b = log_probs_b.reshape(num_envs)
            
                new_env_state = pb_env.step(env_state, action)
                
                r_max = jnp.maximum(r_max, jnp.max(jnp.abs(new_env_state.reward)))
                new_active = jnp.where(new_env_state.done, False, active)
                
                new_carry = (new_env_state, r_max, new_active)
                new_transition = (obs, action, log_probs_b, new_env_state.reward, active)
                
                return new_carry, new_transition
            
            step_keys = jax.random.split(k_steps, pb_rollout_steps)
            final_carry, batch_trajectories = jax.lax.scan(
                step_fn,
                (env_state, r_max, jnp.ones((num_envs,), dtype=bool)),
                step_keys
            )
            
            _, r_max_new, _ = final_carry 
            
            states_t, actions_t, log_probs_t, rewards_t, masks_t = batch_trajectories
            
            states_t    = jnp.swapaxes(states_t, 0, 1)
            actions_t   = jnp.swapaxes(actions_t, 0, 1)
            log_probs_t = jnp.swapaxes(log_probs_t, 0, 1)
            rewards_t   = jnp.swapaxes(rewards_t, 0, 1)
            masks_t     = jnp.swapaxes(masks_t, 0, 1)
            
            t = jnp.arange(pb_rollout_steps)
            returns = jnp.sum((gamma ** t)[None, :] * masks_t * rewards_t, axis=1)
            
            traj = PBTrajectoryBatch(states=states_t, actions=actions_t, rewards=rewards_t,
                            log_probs_b=log_probs_t, masks=masks_t, returns=returns) 
            
            return r_max_new, traj
        
        outer_keys = jax.random.split(prng_key, batches_needed)
        final_r_max, traj_batches = jax.lax.scan(batch_fn, jnp.float32(0.0), outer_keys)
        
        flat_traj = jax.tree.map(
            lambda x: x.reshape(batches_needed * num_envs, *x.shape[2:]), traj_batches
        )                                         
        return flat_traj, final_r_max
    
    return collect_fn