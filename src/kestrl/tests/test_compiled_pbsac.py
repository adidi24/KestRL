"""Tests for make_compiled_pbsac.

Strategy: use a tiny config (small networks, short rollouts, low update freq)
so tests compile and run quickly. Focus on structural correctness — shapes,
metric keys, carry fields — and verify the PB trigger actually fires and
produces non-zero pac_bayes metrics.
"""

import jax
import jax.numpy as jnp
import pytest
import brax.envs

from kestrl.environments.builders.brax_builder import BraxVectorEnv
from kestrl.algorithms.pbsac.compiled.pbsac import make_compiled_pbsac, CompiledPBSACCarry


# ── Config ────────────────────────────────────────────────────────────────────

ENV_NAME   = "inverted_pendulum"
NUM_ENVS   = 2
NUM_SEEDS  = 3

_BASE_CFG = dict(
    hidden_dims           = (16,),
    activation            = "relu",
    total_timesteps       = 400,
    buffer_size           = 512,
    learning_starts       = 10,
    batch_size            = 8,
    train_freq            = 1,
    gradient_steps        = 1,
    gamma                 = 0.99,
    tau                   = 0.005,
    lr_actor              = 3e-4,
    lr_critic             = 3e-4,
    lr_alpha              = 3e-4,
    autotune_alpha        = True,
    target_update_interval= 1,
    log_interval          = 5,
    # PB-SAC specific
    pac_bayes_active           = True,
    pb_update_freq             = 20,        # fires early for testing
    pb_update_epochs           = 2,
    pb_rollout_trajectories    = 4,
    pb_rollout_steps           = 5,
    pb_policy_samples          = 2,
    pb_posterior_lr            = 3e-4,
    pb_rank                    = 2,
    pb_init_std                = 0.01,
    pb_prior_decay             = 0.99,
    delta                      = 0.1,
    mixing_time                = 1,
    actor_freeze_steps         = 2,
    adaptation_samples         = 4,
    explore_prob_init          = 0.5,
    explore_prob_final         = 0.0,
    explore_prob_decay_duration= 0.5,
    explore_n_samples          = 2,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def brax_env():
    raw = brax.envs.create(ENV_NAME, batch_size=NUM_ENVS)
    return BraxVectorEnv(raw, num_envs=NUM_ENVS)


@pytest.fixture(scope="module")
def fns(brax_env):
    return make_compiled_pbsac(brax_env, _BASE_CFG)


@pytest.fixture(scope="module")
def init_carry(fns):
    return jax.jit(fns.init)(jax.random.PRNGKey(0))


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_returns_carry(init_carry):
    assert isinstance(init_carry, CompiledPBSACCarry)


def test_init_carry_shapes(init_carry):
    assert init_carry.global_step.shape == ()
    assert init_carry.global_step == 0
    assert init_carry.r_max == 0.0
    assert init_carry.actor_frozen == False
    assert init_carry.pb_loss == 0.0
    assert init_carry.explore_prob.shape == ()


# ── step_epoch ────────────────────────────────────────────────────────────────

EXPECTED_METRIC_KEYS = {
    'critic/loss', 'actor/loss',
    'episode/return',
    'pac_bayes/loss', 'pac_bayes/kl_div',
    'pac_bayes/mean_return', 'pac_bayes/lambda',
    'pac_bayes/mixing_time', 'pac_bayes/actor_frozen', 'pac_bayes/r_max',
    # autotune_alpha=True adds these
    'alpha/loss', 'alpha/value',
}

def test_step_epoch_metric_keys(fns, init_carry):
    new_carry, metrics = fns.step_epoch(init_carry)
    assert EXPECTED_METRIC_KEYS == set(metrics.keys()), (
        f"Missing: {EXPECTED_METRIC_KEYS - set(metrics.keys())}\n"
        f"Extra:   {set(metrics.keys()) - EXPECTED_METRIC_KEYS}"
    )


def test_step_epoch_metric_shapes(fns, init_carry):
    _, metrics = fns.step_epoch(init_carry)
    log_interval = _BASE_CFG['log_interval']
    for k, v in metrics.items():
        assert v.shape == (log_interval,), f"{k}: expected ({log_interval},), got {v.shape}"


def test_step_epoch_advances_carry(fns, init_carry):
    new_carry, _ = fns.step_epoch(init_carry)
    expected_step = _BASE_CFG['log_interval'] * _BASE_CFG['train_freq'] * NUM_ENVS
    assert int(new_carry.global_step) == expected_step


# ── PB trigger fires ──────────────────────────────────────────────────────────

def test_pb_trigger_fires_and_updates_metrics(fns, init_carry):
    """Run enough epochs so pb_update_freq is crossed; pac_bayes metrics must go non-zero."""
    # pb_update_freq=20, num_envs=2, train_freq=1, log_interval=5
    # new_step per epoch = 5 * 1 * 2 = 10 env steps
    # need > 20 env steps → 3 epochs minimum
    carry = init_carry
    all_pb_loss = []
    for _ in range(6):
        carry, metrics = fns.step_epoch(carry)
        all_pb_loss.extend(metrics['pac_bayes/loss'].tolist())

    assert any(v != 0.0 for v in all_pb_loss), (
        "pac_bayes/loss was 0 for all steps — PB trigger never fired"
    )


def test_pb_r_max_non_negative(fns, init_carry):
    carry = init_carry
    for _ in range(4):
        carry, _ = fns.step_epoch(carry)
    assert float(carry.r_max) >= 0.0
    assert jnp.isfinite(carry.r_max)


# ── pac_bayes_active=False ────────────────────────────────────────────────────

def test_pac_bayes_inactive_still_has_keys(brax_env):
    """With pac_bayes_active=False pac_bayes/* keys must still be present (zeros)."""
    cfg = {**_BASE_CFG, 'pac_bayes_active': False}
    fns = make_compiled_pbsac(brax_env, cfg)
    carry = jax.jit(fns.init)(jax.random.PRNGKey(1))
    _, metrics = fns.step_epoch(carry)
    for k in ('pac_bayes/loss', 'pac_bayes/kl_div', 'pac_bayes/mean_return', 'pac_bayes/lambda'):
        assert k in metrics, f"Missing key {k} when pac_bayes_active=False"
        assert jnp.all(metrics[k] == 0.0), f"{k} should be 0 when pac_bayes_active=False"


# ── actor_frozen branch ───────────────────────────────────────────────────────

def test_actor_frozen_metrics_have_full_structure(fns, init_carry):
    """Metrics dict must have identical structure regardless of actor_frozen state.
    This catches the lax.cond pytree mismatch bug."""
    carry = init_carry
    metric_key_sets = []
    for _ in range(8):
        carry, metrics = fns.step_epoch(carry)
        metric_key_sets.append(frozenset(metrics.keys()))

    assert len(set(metric_key_sets)) == 1, (
        "Metric keys changed across epochs — likely a lax.cond pytree mismatch"
    )


# ── Multi-seed vmap ───────────────────────────────────────────────────────────

def test_vmap_seeds(fns):
    """Replicate the trainer's train_seeds_live pattern: jit(vmap(init)) then
    jit(vmap(step_epoch)). Verifies shapes and that seeds produce independent runs."""
    keys = jax.random.split(jax.random.PRNGKey(42), NUM_SEEDS)
    all_carries  = jax.jit(jax.vmap(fns.init))(keys)
    vmapped_step = jax.jit(jax.vmap(fns.step_epoch))

    log_interval = _BASE_CFG['log_interval']

    # Run enough epochs for PB to fire in at least one
    for _ in range(6):
        all_carries, metrics = vmapped_step(all_carries)

    # Carry has leading num_seeds axis
    assert all_carries.global_step.shape == (NUM_SEEDS,)
    assert all_carries.pb_loss.shape     == (NUM_SEEDS,)

    # Metrics have shape (num_seeds, log_interval)
    for k, v in metrics.items():
        assert v.shape == (NUM_SEEDS, log_interval), (
            f"{k}: expected ({NUM_SEEDS}, {log_interval}), got {v.shape}"
        )

    # Seeds must diverge — identical critic losses would indicate broken key splitting
    critic_losses = metrics['critic/loss']
    assert not jnp.allclose(critic_losses[0], critic_losses[1]), (
        "Seeds 0 and 1 produced identical critic losses — vmap key splitting may be broken"
    )
