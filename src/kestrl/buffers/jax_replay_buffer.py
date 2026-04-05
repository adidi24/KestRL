from typing import NamedTuple

import jax
import jax.numpy as jnp

from kestrl.buffers import ReplayBufferSamples

class JAXTransition(NamedTuple):
    """Rollout output from lax.scan. Leaves: (train_freq, num_envs, *feature)."""
    observations:      jax.Array   # (train_freq, num_envs, obs_dim)
    actions:   jax.Array   # (train_freq, num_envs, act_dim)
    next_observations: jax.Array   # (train_freq, num_envs, obs_dim)
    rewards:   jax.Array   # (train_freq, num_envs)
    dones:     jax.Array   # (train_freq, num_envs)

class JAXBufferState(NamedTuple):
    """Ring-buffer pytree. Leaves: (rows, num_envs, *feature), rows = buffer_size // num_envs."""
    observations:      jax.Array   # (rows, num_envs, obs_dim)
    actions:  jax.Array   # (rows, num_envs, act_dim)
    next_observations: jax.Array   # (rows, num_envs, obs_dim)
    rewards:  jax.Array   # (rows, num_envs)
    dones:    jax.Array   # (rows, num_envs)

def _empty_buffer(rows: int, num_envs: int, obs_dim: int, act_dim: int) -> JAXBufferState:
    """Allocate a zero-initialised JAXBufferState on the default JAX device.

    Args:
        rows    : Number of ring-buffer rows (= buffer_size // num_envs).
        num_envs: Number of parallel environments.
        obs_dim : Flat observation dimension.
        act_dim : Flat action dimension.

    Returns:
        JAXBufferState with all leaves zero-initialised on device.
    """
    row_env = (rows, num_envs)
    return JAXBufferState(
        observations = jnp.zeros((*row_env, obs_dim), dtype=jnp.float32),
        next_observations = jnp.zeros((*row_env, obs_dim), dtype=jnp.float32),
        actions = jnp.zeros((*row_env, act_dim), dtype=jnp.float32),
        rewards = jnp.zeros(row_env, dtype=jnp.float32),
        dones = jnp.zeros(row_env, dtype=jnp.float32),
    )

def _buffer_add(
    buf: JAXBufferState,
    pos: jax.Array,
    full: jax.Array,
    trajectory: JAXTransition,
    rows: int,
) -> tuple[JAXBufferState, jax.Array, jax.Array]:
    """Write a (train_freq, num_envs, ...) trajectory into the ring buffer.

    Computes write indices modulo rows and performs one scatter per stored
    array. Safe to call from inside lax.scan (all ops are JAX-traceable).

    Args:
        buf       : Current buffer state.
        pos       : int32 scalar — next write row (JAX value, part of scan carry).
        full      : bool scalar  — True once every row has been written at least once.
        trajectory: JAXTransition with leaves (train_freq, num_envs, ...).
        rows      : Static Python int — total number of rows in the buffer.

    Returns:
        (new_buf, new_pos, new_full)
    """
    train_freq = trajectory.observations.shape[0]   # static at trace time
    indices    = (jnp.arange(train_freq) + pos) % rows

    new_buf = JAXBufferState(
        observations = buf.observations.at[indices].set(trajectory.observations),
        next_observations = buf.next_observations.at[indices].set(trajectory.next_observations),
        actions = buf.actions.at[indices].set(trajectory.action),
        rewards = buf.rewards.at[indices].set(trajectory.reward),
        dones = buf.dones.at[indices].set(trajectory.done),
    )
    new_pos  = (pos + train_freq) % rows
    new_full = full | ((pos + train_freq) >= rows)
    return new_buf, new_pos, new_full

def _buffer_sample(
      buf: JAXBufferState,
      rng: jax.Array,
      batch_size: int,
      upper: jax.Array,          # jnp.where(full, rows, pos) effective buffer size
  ) -> ReplayBufferSamples: 
    """
    Sample a batch of transitions from the replay buffer.

    Args:
        buf: Current replay buffer state.
        rng: JAX random key.
        batch_size: Number of transitions to sample.
        upper: Effective buffer size.

    Returns:
        ReplayBufferSamples.
    """
    k1, k2 = jax.random.split(rng)
    
    row_inds = jax.random.randint(k1, (batch_size,), 0, upper)
    env_inds = jax.random.randint(k2, (batch_size,), 0, buf.observations.shape[1])
    
    return ReplayBufferSamples(
        observations = buf.observations[row_inds, env_inds],
        actions = buf.actions[row_inds, env_inds],
        next_observations = buf.next_observations[row_inds, env_inds],
        rewards = buf.rewards[row_inds, env_inds, None],
        dones = buf.dones[row_inds, env_inds, None],
    )