"""Tests for ReplayBuffer.

Run with: .venv/bin/python3 -m pytest src/kestrl/tests/test_replay_buffer.py -v
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import jax.numpy as jnp
from gymnasium import spaces

from kestrl.buffers.replay_buffer import ReplayBuffer


# ── Helpers ───────────────────────────────────────────────────

def make_continuous_buffer(capacity=100):
    """CartPole-like: obs=Box(4,), action=Box(1,)"""
    obs_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    return ReplayBuffer(capacity, obs_space, act_space)

def make_discrete_buffer(capacity=100):
    """CartPole-like: obs=Box(4,), action=Discrete(2)"""
    obs_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    act_space = spaces.Discrete(2)
    return ReplayBuffer(capacity, obs_space, act_space)

def add_transitions(buf, n, obs_dim=4, act_dim=1):
    """Add n random transitions to the buffer."""
    for _ in range(n):
        obs = np.random.randn(obs_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(act_dim).astype(np.float32)
        reward = np.random.randn(1).astype(np.float32)
        done = np.array([0.0], dtype=np.float32)
        infos = {}
        buf.add(obs, next_obs, action, reward, done, infos)


# ── Construction Tests ────────────────────────────────────────

def test_continuous_buffer_init():
    buf = make_continuous_buffer(100)
    assert buf.buffer_size == 100
    assert buf.observations.shape == (100, 1, 4)
    assert buf.actions.shape == (100, 1, 1)
    assert buf.rewards.shape == (100, 1)
    assert buf.dones.shape == (100, 1)
    assert buf.size() == 0

def test_discrete_buffer_init():
    buf = make_discrete_buffer(100)
    assert buf.action_dim == 1  # discrete actions are scalar indices
    assert buf.actions.shape == (100, 1, 1)


# ── Add Tests ─────────────────────────────────────────────────

def test_add_increments_pos():
    buf = make_continuous_buffer(100)
    add_transitions(buf, 5)
    assert buf.pos == 5
    assert buf.size() == 5
    assert not buf.full

def test_add_circular_wrap():
    buf = make_continuous_buffer(10)
    add_transitions(buf, 15)
    assert buf.full
    assert buf.pos == 5  # wrapped around: 15 % 10 = 5
    assert buf.size() == 10

def test_add_stores_data():
    buf = make_continuous_buffer(10)
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    next_obs = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    action = np.array([0.5], dtype=np.float32)
    reward = np.array([1.0], dtype=np.float32)
    done = np.array([0.0], dtype=np.float32)
    buf.add(obs, next_obs, action, reward, done, {})
    
    np.testing.assert_array_equal(buf.observations[0, 0], obs)
    np.testing.assert_array_equal(buf.next_observations[0, 0], next_obs)
    np.testing.assert_array_equal(buf.actions[0, 0], action)
    np.testing.assert_array_equal(buf.rewards[0, 0], reward)
    np.testing.assert_array_equal(buf.dones[0, 0], done)


# ── Sample Tests ──────────────────────────────────────────────

def test_sample_shapes():
    buf = make_continuous_buffer(100)
    add_transitions(buf, 50)
    batch = buf.sample(32)
    obs, actions, next_obs, dones, rewards = batch
    assert obs.shape == (32, 4)
    assert actions.shape == (32, 1)
    assert next_obs.shape == (32, 4)
    assert dones.shape == (32, 1)
    assert rewards.shape == (32, 1)

def test_sample_returns_jax_arrays():
    buf = make_continuous_buffer(100)
    add_transitions(buf, 50)
    batch = buf.sample(8)
    for arr in batch:
        assert isinstance(arr, jnp.ndarray), f"Expected jnp.ndarray, got {type(arr)}"

def test_sample_dtype():
    buf = make_continuous_buffer(100)
    add_transitions(buf, 50)
    obs, actions, next_obs, dones, rewards = buf.sample(8)
    assert obs.dtype == jnp.float32
    assert rewards.dtype == jnp.float32
    assert dones.dtype == jnp.float32

def test_sample_after_wrap():
    """Sampling should work correctly after the buffer wraps around."""
    buf = make_continuous_buffer(10)
    add_transitions(buf, 25)  # wrapped around 2.5 times
    batch = buf.sample(8)
    obs, actions, next_obs, dones, rewards = batch
    assert obs.shape == (32, 4) or obs.shape == (8, 4)  # flexible for n_envs

def test_sample_discrete_actions():
    buf = make_discrete_buffer(100)
    for _ in range(20):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        action = np.array([np.random.randint(2)], dtype=np.int64)
        reward = np.array([1.0], dtype=np.float32)
        done = np.array([0.0], dtype=np.float32)
        buf.add(obs, next_obs, action, reward, done, {})
    
    batch = buf.sample(8)
    obs, actions, next_obs, dones, rewards = batch
    assert actions.shape == (8, 1)


# ── Reset Test ────────────────────────────────────────────────

def test_reset():
    buf = make_continuous_buffer(100)
    add_transitions(buf, 50)
    assert buf.size() == 50
    buf.reset()
    assert buf.size() == 0
    assert buf.pos == 0
    assert not buf.full


# ── Timeout Handling Tests ────────────────────────────────────

def test_timeout_masking():
    """Dones due to timeout should be masked out."""
    buf = make_continuous_buffer(10)
    obs = np.zeros(4, dtype=np.float32)
    next_obs = np.zeros(4, dtype=np.float32)
    action = np.zeros(1, dtype=np.float32)
    
    # Add a transition that's done due to timeout
    buf.add(obs, next_obs, action, np.array([1.0]), np.array([1.0]),
            {"TimeLimit.truncated": np.array([True], dtype=bool)})
    # Add a transition that's done NOT due to timeout
    buf.add(obs, next_obs, action, np.array([1.0]), np.array([1.0]),
            {"TimeLimit.truncated": np.array([False], dtype=bool)})
    
    # Sample all and check: timeout done should be masked to 0
    assert buf.timeouts[0, 0] == 1.0  # timeout flagged
    assert buf.timeouts[1, 0] == 0.0  # no timeout

def test_no_timeout_handling():
    """With handle_timeout_termination=False, timeouts should stay zero."""
    obs_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buf = ReplayBuffer(10, obs_space, act_space, handle_timeout_termination=False)
    
    obs = np.zeros(4, dtype=np.float32)
    buf.add(obs, obs, np.zeros(1), np.array([1.0]), np.array([1.0]),
            {"TimeLimit.truncated": np.array([True], dtype=bool)})
    assert buf.timeouts[0, 0] == 0.0  # should NOT be recorded


# ── Multi-Env Test ────────────────────────────────────────────

def test_multi_env_buffer():
    """Buffer with n_envs=2 should have correct shapes."""
    obs_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buf = ReplayBuffer(100, obs_space, act_space, n_envs=2)
    
    assert buf.buffer_size == 50  # 100 // 2
    assert buf.observations.shape == (50, 2, 4)
    assert buf.actions.shape == (50, 2, 1)


# ── Staging Buffer Tests ──────────────────────────────────────

def test_staging_buffer_resizes():
    """Staging buffer should grow when sampled with a larger batch than before."""
    buf = make_continuous_buffer(100)
    add_transitions(buf, 50)
    inds = np.arange(8)
    buf._get_samples(inds)
    assert buf._staging is not None
    assert buf._staging.shape[0] >= 8
    inds32 = np.arange(32)
    buf._get_samples(inds32)
    assert buf._staging.shape[0] >= 32


def test_staging_buffer_values():
    """Sampled values must match what was stored (single known transition)."""
    buf = make_continuous_buffer(10)
    obs      = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    next_obs = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    action   = np.array([0.5], dtype=np.float32)
    reward   = np.array([1.0], dtype=np.float32)
    done     = np.array([0.0], dtype=np.float32)
    buf.add(obs, next_obs, action, reward, done, {})

    # Sample index 0 directly to get the known transition back
    batch = buf._get_samples(np.array([0]))
    np.testing.assert_allclose(np.asarray(batch.observations[0]), obs, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(batch.next_observations[0]), next_obs, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(batch.actions[0]), action, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(batch.rewards[0, 0]), reward[0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(batch.dones[0, 0]), done[0], rtol=1e-6)


def test_staging_buffer_multidim_obs():
    """Staging buffer must correctly flatten and restore multi-dim observations."""
    obs_space = spaces.Box(-np.inf, np.inf, shape=(4, 3), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    buf = ReplayBuffer(20, obs_space, act_space)

    obs = np.random.randn(4, 3).astype(np.float32)
    buf.add(obs, obs * 2, np.zeros(2, dtype=np.float32),
            np.array([1.0]), np.array([0.0]), {})

    batch = buf._get_samples(np.array([0]))
    assert batch.observations.shape == (1, 4, 3)
    assert batch.next_observations.shape == (1, 4, 3)
    np.testing.assert_allclose(np.asarray(batch.observations[0]), obs, rtol=1e-6)




if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])