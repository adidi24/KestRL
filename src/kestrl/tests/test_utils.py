"""Tests for utility functions.

Run with: .venv/bin/python3 -m pytest src/kestrl/tests/test_utils.py -v
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.utils import set_seed, soft_update, linear_schedule
from kestrl.networks import MLP


# ── set_seed ──────────────────────────────────────────────────

def test_set_seed_returns_key():
    key = set_seed(42)
    assert key.shape == (2,) or key.shape == ()  # depends on JAX version

def test_set_seed_deterministic():
    k1 = set_seed(42)
    k2 = set_seed(42)
    assert jnp.array_equal(k1, k2)

def test_set_seed_different_seeds():
    k1 = set_seed(0)
    k2 = set_seed(1)
    assert not jnp.array_equal(k1, k2)


# ── soft_update ───────────────────────────────────────────────

def test_soft_update_tau_one():
    """tau=1.0 should fully copy online → target."""
    m1 = MLP(4, (8,), 2, rngs=nnx.Rngs(0))
    m2 = MLP(4, (8,), 2, rngs=nnx.Rngs(1))
    x = jnp.ones((1, 4))
    soft_update(m1, m2, tau=1.0)
    assert jnp.allclose(m1(x), m2(x))

def test_soft_update_tau_zero():
    """tau=0.0 should NOT change target."""
    m1 = MLP(4, (8,), 2, rngs=nnx.Rngs(0))
    m2 = MLP(4, (8,), 2, rngs=nnx.Rngs(1))
    x = jnp.ones((1, 4))
    before = m2(x).copy()
    soft_update(m1, m2, tau=0.0)
    assert jnp.allclose(before, m2(x))

def test_soft_update_interpolates():
    """tau=0.5 should give params halfway between online and target."""
    m1 = MLP(4, (8,), 2, rngs=nnx.Rngs(0))
    m2 = MLP(4, (8,), 2, rngs=nnx.Rngs(1))
    # Snapshot values BEFORE update (nnx.state returns a live reference)
    s1_vals = jax.tree.map(lambda x: x.copy(), nnx.state(m1, nnx.Param))
    s2_vals = jax.tree.map(lambda x: x.copy(), nnx.state(m2, nnx.Param))
    soft_update(m1, m2, tau=0.5)
    s2_after = nnx.state(m2, nnx.Param)
    # Check that each param is the average
    expected = jax.tree.map(lambda a, b: 0.5 * a + 0.5 * b, s2_vals, s1_vals)
    match = jax.tree.map(lambda a, b: jnp.allclose(a, b), s2_after, expected)
    assert all(jax.tree.leaves(match))


# ── linear_schedule ───────────────────────────────────────────

def test_linear_schedule_start():
    assert linear_schedule(1.0, 0.1, 100, 0) == 1.0

def test_linear_schedule_end():
    assert linear_schedule(1.0, 0.1, 100, 100) == 0.1

def test_linear_schedule_midpoint():
    val = linear_schedule(1.0, 0.1, 100, 50)
    assert jnp.isclose(val, 0.55)

def test_linear_schedule_clamps():
    """After duration, should clamp at end value."""
    assert linear_schedule(1.0, 0.1, 100, 200) == 0.1


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
