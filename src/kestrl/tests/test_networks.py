"""Tests for MLP and MultiHeadMLP networks.

Run with: .venv/bin/python3 -m pytest src/kestrl/tests/test_networks.py -v
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.networks import MLP
from kestrl.networks.multi_head_mlp import MultiHeadMLP


# ── MLP Tests ──────────────────────────────────────────────────

def test_mlp_basic_shape():
    model = MLP(in_dim=4, hidden_dims=(256, 256), out_dim=2, rngs=nnx.Rngs(0))
    y = model(jnp.ones((1, 4)))
    assert y.shape == (1, 2)

def test_mlp_batch():
    model = MLP(in_dim=4, hidden_dims=(256, 256), out_dim=2, rngs=nnx.Rngs(0))
    y = model(jnp.ones((32, 4)))
    assert y.shape == (32, 2)

def test_mlp_tanh_activation():
    model = MLP(in_dim=8, hidden_dims=(64,), out_dim=1, activation='tanh', rngs=nnx.Rngs(42))
    y = model(jnp.ones((1, 8)))
    assert y.shape == (1, 1)

def test_mlp_params_inspectable():
    model = MLP(in_dim=4, hidden_dims=(256, 256), out_dim=2, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    shapes = jax.tree.map(lambda x: x.shape, params)
    assert shapes is not None

def test_mlp_jit():
    model = MLP(in_dim=4, hidden_dims=(256, 256), out_dim=2, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 4))
    y_eager = model(x)

    @nnx.jit
    def forward(m, x):
        return m(x)

    y_jit = forward(model, x)
    assert jnp.allclose(y_eager, y_jit)


# ── MultiHeadMLP Tests ────────────────────────────────────────

def test_multi_head_actor_critic():
    model = MultiHeadMLP(
        in_dim=4, hidden_dims=(256, 256),
        head_configs={'actor': 2, 'critic': 1},
        rngs=nnx.Rngs(0),
    )
    out = model(jnp.ones((1, 4)))
    assert isinstance(out, dict)
    assert out['actor'].shape == (1, 2)
    assert out['critic'].shape == (1, 1)

def test_multi_head_batch():
    model = MultiHeadMLP(
        in_dim=4, hidden_dims=(256, 256),
        head_configs={'actor': 2, 'critic': 1},
        rngs=nnx.Rngs(0),
    )
    out = model(jnp.ones((32, 4)))
    assert out['actor'].shape == (32, 2)
    assert out['critic'].shape == (32, 1)

def test_multi_head_per_head_activation():
    model = MultiHeadMLP(
        in_dim=8, hidden_dims=(64, 64),
        head_configs={'mean': 3, 'log_std': 3, 'value': 1},
        head_activations={'mean': 'tanh'},
        rngs=nnx.Rngs(42),
    )
    out = model(jnp.ones((1, 8)))
    assert out['mean'].shape == (1, 3)
    assert out['log_std'].shape == (1, 3)
    assert out['value'].shape == (1, 1)
    # tanh bounds
    assert jnp.all(out['mean'] >= -1.0) and jnp.all(out['mean'] <= 1.0)

def test_multi_head_jit():
    model = MultiHeadMLP(
        in_dim=4, hidden_dims=(256, 256),
        head_configs={'actor': 2, 'critic': 1},
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((1, 4))
    out_eager = model(x)

    @nnx.jit
    def forward(m, x):
        return m(x)

    out_jit = forward(model, x)
    assert jnp.allclose(out_eager['actor'], out_jit['actor'])
    assert jnp.allclose(out_eager['critic'], out_jit['critic'])

def test_mlp_uses_multi_head():
    """MLP should be a thin wrapper around MultiHeadMLP."""
    mlp = MLP(in_dim=4, hidden_dims=(64,), out_dim=3, rngs=nnx.Rngs(0))
    y = mlp(jnp.ones((1, 4)))
    assert y.shape == (1, 3)
    assert hasattr(mlp, '_net')
    assert isinstance(mlp._net, MultiHeadMLP)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
