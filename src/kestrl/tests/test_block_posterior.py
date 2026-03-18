"""
Tests for src/kestrl/distributions/block_posterior.py   (task 5.1f)

Run with:
    uv run pytest src/kestrl/tests/test_block_posterior.py -v

Coverage:
    T1  from_actor builds correct structure
    T2  from_posterior gives KL = 0 at init
    T3  P=0 block KL matches closed-form diagonal KL
    T4  KL >= 0 for 50 random configs
    T5  jax.jit(kl_block) compiles and runs correctly
    T6  nnx.value_and_grad + nnx.Optimizer work on posterior (needed for REINFORCE)
    T7  sample mean and covariance match theoretical values
"""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from kestrl.distributions import (
    LayerPosterior,
    BlockPosterior,
    BlockPrior,
    block_sample,
    kl_block,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class TinyActor(nnx.Module):
    """Minimal two-layer MLP for testing without depending on KestRL networks."""
    def __init__(self):
        self.dense1 = nnx.Linear(4, 8,  rngs=nnx.Rngs(0))
        self.dense2 = nnx.Linear(8, 2,  rngs=nnx.Rngs(0))


RANK      = 3
INIT_STD  = 0.01

@pytest.fixture(scope="module")
def actor():
    return TinyActor()

@pytest.fixture(scope="module")
def posterior(actor):
    return BlockPosterior.from_actor(actor, rank=RANK, init_std=INIT_STD)

@pytest.fixture(scope="module")
def prior(posterior):
    return BlockPrior.from_posterior(posterior)


# ---------------------------------------------------------------------------
# T1 — from_actor builds the correct structure
# ---------------------------------------------------------------------------

def test_from_actor_structure(actor, posterior):
    expected_keys = {('dense1', 'bias'), ('dense1', 'kernel'), ('dense2', 'bias'), ('dense2', 'kernel')}
    assert set(posterior.layers.keys()) == expected_keys, (
        f"Unexpected layer names: {set(posterior.layers.keys())}"
    )
    assert posterior.rank == RANK

    for name, lp in posterior.layers.items():
        assert lp.mean.get_value().ndim == 1,                                f"{name}: mean must be flat"
        assert lp.std.shape == lp.mean.get_value().shape,        f"{name}: std shape mismatch"
        assert lp.P.get_value().shape   == (lp.mean.get_value().shape[0], RANK), f"{name}: P shape mismatch"
        assert jnp.all(lp.P.get_value() == 0),                               f"{name}: P must be zeros at init"
        assert jnp.allclose(lp.std, INIT_STD),                   f"{name}: std must equal init_std"


# ---------------------------------------------------------------------------
# T2 — from_posterior gives KL = 0 at init
# ---------------------------------------------------------------------------

def test_kl_zero_at_init(posterior, prior):
    kl = kl_block(posterior, prior)
    assert abs(float(kl)) < 1e-5, f"KL at init should be 0, got {kl:.6f}"


# ---------------------------------------------------------------------------
# T3 — P=0 block KL matches closed-form diagonal KL
# ---------------------------------------------------------------------------

def test_p_zero_matches_diagonal_kl():
    rng = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    D = 32

    mean_q = jax.random.normal(k1, (D,))
    std_q  = jnp.abs(jax.random.normal(k2, (D,))) + 0.1
    mean_p = jax.random.normal(k3, (D,))
    std_p  = jnp.abs(jax.random.normal(k4, (D,))) + 0.1

    # BlockPosterior with P = 0
    lp     = LayerPosterior(mean=mean_q, std=std_q, P=jnp.zeros((D, 5)))
    post   = BlockPosterior(layers={'only': lp}, rank=5)
    pr     = BlockPrior(layers={'only': (mean_p, std_p)})
    block_kl_val = kl_block(post, pr)

    # Reference: closed-form diagonal KL
    diag_kl = 0.5 * jnp.sum(
        2 * (jnp.log(std_p) - jnp.log(std_q))
        + (std_q ** 2 + (mean_q - mean_p) ** 2) / std_p ** 2
        - 1.0
    )

    assert jnp.allclose(block_kl_val, diag_kl, atol=1e-5), (
        f"P=0 mismatch: block={block_kl_val:.6f}, diagonal={diag_kl:.6f}"
    )


# ---------------------------------------------------------------------------
# T4 — KL >= 0 over 50 random configurations
# ---------------------------------------------------------------------------

def test_kl_nonneg():
    D, K = 20, 4
    failures = []
    for seed in range(50):
        rng = jax.random.PRNGKey(seed)
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

        lp   = LayerPosterior(
            mean=jax.random.normal(k1, (D,)),
            std =jnp.abs(jax.random.normal(k2, (D,))) + 0.05,
            P   =jax.random.normal(k3, (D, K)) * 0.1,
        )
        post = BlockPosterior(layers={'l': lp}, rank=K)
        pr   = BlockPrior(layers={'l': (
            jax.random.normal(k4, (D,)),
            jnp.abs(jax.random.normal(k5, (D,))) + 0.05,
        )})

        kl = float(kl_block(post, pr))
        if kl < -1e-5:
            failures.append((seed, kl))

    assert not failures, f"KL was negative for seeds: {failures}"


# ---------------------------------------------------------------------------
# T5 — jax.jit(kl_block) compiles and agrees with eager mode
# ---------------------------------------------------------------------------

def test_kl_jit_compatible(posterior, prior):
    kl_jit = jax.jit(kl_block)
    # Warmup (triggers compilation)
    kl_eager = float(kl_block(posterior, prior))
    kl_jitted = float(kl_jit(posterior, prior))
    assert abs(kl_eager - kl_jitted) < 1e-4, (
        f"JIT result differs from eager: {kl_eager} vs {kl_jitted}"
    )


# ---------------------------------------------------------------------------
# T6 — nnx.value_and_grad + nnx.Optimizer work on posterior (critical for REINFORCE)
# ---------------------------------------------------------------------------

def test_kl_grad_wrt_posterior():
    """Validates the exact optimizer pattern used in task 5.4 (REINFORCE update):

        loss_val, grads = nnx.value_and_grad(loss_fn)(posterior, prior)
        optimizer.update(grads)
    """
    import optax
    D, K = 12, 2
    rng = jax.random.PRNGKey(99)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    lp   = LayerPosterior(
        mean=jax.random.normal(k1, (D,)),
        std =jnp.abs(jax.random.normal(k2, (D,))) + 0.1,
        P   =jax.random.normal(k3, (D, K)) * 0.05,
    )
    post = BlockPosterior(layers={'l': lp}, rank=K)
    pr   = BlockPrior(layers={'l': (
        jax.random.normal(k4, (D,)),
        jnp.abs(jax.random.normal(k5, (D,))) + 0.1,
    )})

    # nnx.value_and_grad differentiates w.r.t. nnx.Param fields in the first arg
    loss_val, grads = nnx.value_and_grad(kl_block)(post, pr)
    assert jnp.isfinite(loss_val), "KL loss must be finite"

    # Verify an optimizer step actually runs
    mean_before = post.layers['l'].mean.get_value().copy()
    optimizer   = nnx.Optimizer(post, optax.adam(1e-3), wrt=nnx.Param)
    optimizer.update(post, grads)
    mean_after  = post.layers['l'].mean.get_value()
    assert not jnp.allclose(mean_before, mean_after), (
        "Optimizer update should have changed the posterior mean"
    )


# ---------------------------------------------------------------------------
# T7 — sample mean and covariance match theoretical values
# ---------------------------------------------------------------------------

def test_sample_statistics():
    D, K = 16, 3
    mean = jnp.ones(D) * 0.5
    std  = jnp.ones(D) * 0.2
    P    = jnp.full((D, K), 0.1)  # known low-rank factor

    lp = LayerPosterior(mean=mean, std=std, P=P)

    # Draw N samples via vmap over keys
    N = 20_000
    keys = jax.random.split(jax.random.PRNGKey(42), N)
    samples = jax.vmap(lp.sample)(keys)  # (N, D)
    mean = lp.mean.get_value()  # unwrap after construction for comparison

    # Empirical mean ≈ v
    emp_mean = jnp.mean(samples, axis=0)
    assert jnp.allclose(emp_mean, mean, atol=0.02), (
        f"Mean mismatch: max_err={jnp.max(jnp.abs(emp_mean - mean)):.4f}"
    )

    # Empirical covariance ≈ P Pᵀ + diag(σ²)
    expected_cov = P @ P.T + jnp.diag(std ** 2)
    centered = samples - emp_mean
    emp_cov = (centered.T @ centered) / (N - 1)
    cov_err = jnp.max(jnp.abs(emp_cov - expected_cov))
    assert cov_err < 0.05, f"Covariance mismatch: max_err={cov_err:.4f}"
