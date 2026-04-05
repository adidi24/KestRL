import jax
from flax import nnx

from kestrl.algorithms.sac.functions import (
    get_continuous_actor_action,
    get_discrete_actor_action,
)
from kestrl.buffers import JAXTransition

def _make_jax_scan_rollout(
    bundle_gd,
    env,
    is_discrete: bool,
    action_scale: jax.Array | None,
    action_bias: jax.Array | None
):
    """Return a JIT'd scan-based rollout function for a Brax/MJX environment.

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
        def _act(actor, obs, key):
            action, _, _ = get_discrete_actor_action(actor, obs, key)
            return action
    else:
        assert action_scale is not None and action_bias is not None, "action_scale and action_bias must be provided for continuous environments"
        def _act(actor, obs, key):
            action, _, _ = get_continuous_actor_action(
                actor, obs, action_scale, action_bias, key
            )
            return action
    
    def collect_rollout(bundle_state, jax_env_state, prng_keys):
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
        actor = b.actor
        
        def step_fn(env_state, key):
            obs = env_state.obs
            action = _act(actor, obs, key)
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
    