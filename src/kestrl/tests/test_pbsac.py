"""Tests for PB-SAC pure functions and frozen-bundle update functions.

Coverage:
  - compute_policy_is_return
  - compute_pac_bayes_loss  (vmap refactor)
  - compute_pac_bayes_bound (vmap refactor, pending)
  - _make_frozen_update_pb_posterior
  - _make_frozen_sync_posterior / _make_frozen_inject_posterior
  - _make_frozen_adaptive_continuous_update / _make_frozen_adaptive_discrete_update
  - _estimate_mixing_time (numpy path, standard route)
  - explore_prob decay formula
"""

import copy

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from kestrl.networks.mlp import MLP
from kestrl.networks.multi_head_mlp import MultiHeadMLP
from kestrl.distributions import (
    BlockPosterior,
    BlockPrior,
)
from kestrl.algorithms.pbsac.functions import (
    compute_policy_is_return,
    compute_pac_bayes_loss,
    compute_pac_bayes_bound,
)
from kestrl.algorithms.pbsac.updates import (
    _make_frozen_update_pb_posterior,
    _make_frozen_sync_posterior,
    _make_frozen_inject_posterior,
    _make_frozen_adaptive_continuous_update,
    _make_frozen_adaptive_discrete_update,
)
from kestrl.algorithms.pbsac.pbsac import _PBSACBundle, PBSAC, PBTrajectory
from kestrl.algorithms.sac.sac import LogAlpha


# ── Shared fixtures ───────────────────────────────────────────────────────────

OBS_DIM  = 4
ACT_DIM  = 2
N_TRAJ   = 8    # number of trajectories in a batch
H        = 10   # trajectory horizon


def _make_batch_data(obs_dim=OBS_DIM, act_dim=ACT_DIM, n_traj=N_TRAJ, H=H,
                     is_discrete=False, seed=0):
    """Minimal batch_data dict for PAC-Bayes function tests."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    data = {
        'states':      jax.random.normal(k1, (n_traj, H, obs_dim)),
        'log_probs_b': jax.random.normal(k2, (n_traj, H)) * 0.1,
        'masks':       jnp.ones((n_traj, H), dtype=bool),
        'returns':     jax.random.uniform(k3, (n_traj,)),
    }
    if is_discrete:
        data['actions'] = jax.random.randint(key, (n_traj, H), 0, act_dim)
    else:
        data['actions']       = jax.random.normal(key, (n_traj, H, act_dim)) * 0.1
        data['action_scale']  = jnp.ones(act_dim)
        data['action_bias']   = jnp.zeros(act_dim)
    return data


def _make_continuous_actor(obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0):
    return MultiHeadMLP(
        obs_dim, {'mean': act_dim, 'log_std': act_dim}, [16],
        rngs=nnx.Rngs(seed),
    )


def _make_bundle_and_split(obs_dim=OBS_DIM, act_dim=ACT_DIM, rank=2,
                            init_std=0.01, autotune_alpha=True, seed=0):
    """Build a minimal continuous _PBSACBundle and return (graphdef, state)."""
    actor   = _make_continuous_actor(obs_dim, act_dim, seed)
    critic1 = MLP(obs_dim + act_dim, 1, [16], rngs=nnx.Rngs(seed + 1))
    critic2 = MLP(obs_dim + act_dim, 1, [16], rngs=nnx.Rngs(seed + 2))

    b = _PBSACBundle()
    b.actor        = actor
    b.actor_opt    = nnx.Optimizer(actor, optax.adam(1e-3), wrt=nnx.Param)
    b.critic1      = critic1
    b.critic1_opt  = nnx.Optimizer(critic1, optax.adam(1e-3), wrt=nnx.Param)
    b.critic2      = critic2
    b.critic2_opt  = nnx.Optimizer(critic2, optax.adam(1e-3), wrt=nnx.Param)
    b.target_critic1 = copy.deepcopy(critic1)
    b.target_critic2 = copy.deepcopy(critic2)
    b.log_alpha    = LogAlpha(0.0)
    if autotune_alpha:
        b.alpha_opt = nnx.Optimizer(b.log_alpha, optax.adam(1e-3), wrt=nnx.Param)
    b.posterior    = BlockPosterior.from_actor(actor, rank=rank, init_std=init_std)
    b.prior        = BlockPrior.from_posterior(b.posterior)
    b.pb_optimizer = nnx.Optimizer(b.posterior, optax.adam(1e-3), wrt=nnx.Param)

    gd, state = nnx.split(b)
    return gd, state


# ── compute_policy_is_return ─────────────────────────────────────────────────

def test_policy_is_return_scalar_and_finite():
    actor = _make_continuous_actor()
    data  = _make_batch_data()
    result = compute_policy_is_return(actor, is_discrete=False, batch_data=data)
    assert result.shape == (), f"Expected scalar, got {result.shape}"
    assert jnp.isfinite(result), "IS return is not finite"


def test_policy_is_return_mask_zeros_ignored():
    """Masking out all but the first step should change the IS weight computation."""
    actor      = _make_continuous_actor()
    data_full  = _make_batch_data()
    data_sparse = {**data_full, 'masks': jnp.zeros_like(data_full['masks']).at[:, 0].set(True)}

    r_full   = compute_policy_is_return(actor, is_discrete=False, batch_data=data_full)
    r_sparse = compute_policy_is_return(actor, is_discrete=False, batch_data=data_sparse)
    assert not jnp.allclose(r_full, r_sparse), (
        "Full mask and sparse mask should produce different IS returns"
    )


def test_policy_is_return_discrete():
    obs_dim, n_actions = 4, 3
    actor  = MLP(obs_dim, n_actions, [16], rngs=nnx.Rngs(0))
    data   = _make_batch_data(obs_dim=obs_dim, act_dim=n_actions, is_discrete=True)
    result = compute_policy_is_return(actor, is_discrete=True, batch_data=data)
    assert result.shape == ()
    assert jnp.isfinite(result)


# ── compute_pac_bayes_loss ────────────────────────────────────────────────────

def _make_loss_inputs(init_std=0.01, num_samples=4, seed=0):
    actor     = _make_continuous_actor(seed=seed)
    posterior = BlockPosterior.from_actor(actor, rank=2, init_std=init_std)
    prior     = BlockPrior.from_posterior(posterior)
    data      = _make_batch_data(seed=seed)
    return actor, posterior, prior, data, num_samples


def test_pac_bayes_loss_output_structure():
    actor, posterior, prior, data, num_samples = _make_loss_inputs()
    key  = jax.random.PRNGKey(0)
    loss, metrics = compute_pac_bayes_loss(
        posterior, prior, actor, is_discrete=False,
        key=key, num_samples=num_samples,
        lambda_val=1.0, C_const=1.0, C_prime_const=0.1,
        batch_data=data,
    )
    assert loss.shape == (), f"Loss must be scalar, got {loss.shape}"
    assert jnp.isfinite(loss), "Loss is not finite"
    for key_name in ('loss', 'kl_div', 'mean_empirical_return', 'lambda'):
        assert key_name in metrics, f"Missing key '{key_name}' in metrics"


def test_pac_bayes_loss_kl_zero_at_init():
    """KL(posterior || prior) == 0 when posterior was used to init the prior."""
    actor, posterior, prior, data, _ = _make_loss_inputs(init_std=0.01)
    _, metrics = compute_pac_bayes_loss(
        posterior, prior, actor, is_discrete=False,
        key=jax.random.PRNGKey(1), num_samples=2,
        lambda_val=1.0, C_const=0.0, C_prime_const=0.0,
        batch_data=data,
    )
    # KL should be exactly 0 at init (prior == posterior by construction)
    assert float(metrics['kl_div']) < 1e-5, (
        f"KL should be ~0 at init, got {metrics['kl_div']:.6f}"
    )


def test_pac_bayes_loss_finite_for_multiple_samples():
    actor, posterior, prior, data, _ = _make_loss_inputs(init_std=0.1, num_samples=8)
    loss, metrics = compute_pac_bayes_loss(
        posterior, prior, actor, is_discrete=False,
        key=jax.random.PRNGKey(7), num_samples=8,
        lambda_val=1.0, C_const=1.0, C_prime_const=0.1,
        batch_data=data,
    )
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(v) for v in metrics.values())


# ── compute_pac_bayes_bound ───────────────────────────────────────────────────

def test_pac_bayes_bound_output_structure():
    actor, posterior, prior, data, _ = _make_loss_inputs()
    result = compute_pac_bayes_bound(
        posterior, prior, actor, is_discrete=False,
        key=jax.random.PRNGKey(0), num_samples=4,
        T=100, H=H, r_max=1.0, mixing_time=1,
        gamma=0.99, delta=0.05, batch_data=data,
    )
    for k in ('certified_return', 'empirical_return', 'uncertainty_term',
              'kl_div', 'c_squared', 'mixing_time', 'r_max'):
        assert k in result, f"Missing key '{k}'"
    assert all(jnp.isfinite(v) for v in result.values())


def test_pac_bayes_bound_certified_below_empirical():
    actor, posterior, prior, data, _ = _make_loss_inputs()
    result = compute_pac_bayes_bound(
        posterior, prior, actor, is_discrete=False,
        key=jax.random.PRNGKey(0), num_samples=4,
        T=100, H=H, r_max=1.0, mixing_time=1,
        gamma=0.99, delta=0.05, batch_data=data,
    )
    assert float(result['certified_return']) < float(result['empirical_return']), (
        "Certified return must be strictly below empirical return"
    )
    assert float(result['uncertainty_term']) > 0.0
    assert float(result['kl_div']) >= 0.0


# ── Frozen-bundle update functions ────────────────────────────────────────────

def test_frozen_update_pb_posterior_changes_state():
    """One posterior gradient step must change bundle_state."""
    gd, state = _make_bundle_and_split()
    data  = _make_batch_data()
    update_fn, _ = _make_frozen_update_pb_posterior(
        gd, is_discrete=False, num_samples=2
    )

    new_state, metrics = update_fn(
        state, C_const=1.0, C_prime_const=0.1, batch_data=data,
        key=jax.random.PRNGKey(0),
    )

    # States must differ (posterior params were updated)
    orig_leaves = jax.tree.leaves(state)
    new_leaves  = jax.tree.leaves(new_state)
    any_changed = any(
        not jnp.array_equal(o, n)
        for o, n in zip(orig_leaves, new_leaves)
    )
    assert any_changed, "bundle_state unchanged after posterior gradient step"
    for k in ('loss', 'kl_div', 'mean_empirical_return', 'lambda'):
        assert k in metrics


def test_frozen_update_pb_posterior_jit_stable():
    """Calling the update function twice must not error (JIT cache hit)."""
    gd, state = _make_bundle_and_split()
    data = _make_batch_data()
    update_fn, _ = _make_frozen_update_pb_posterior(gd, is_discrete=False, num_samples=2)
    key = jax.random.PRNGKey(0)
    state, _ = update_fn(state, 1.0, 0.1, data, key)
    state, _ = update_fn(state, 1.0, 0.1, data, key)  # must not retrace


def test_frozen_sync_posterior_copies_actor_weights():
    """After sync, each posterior layer mean must equal the flattened actor param."""
    gd, state = _make_bundle_and_split()
    sync_fn = _make_frozen_sync_posterior(gd)
    new_state = sync_fn(state)

    b = nnx.merge(gd, new_state)
    actor_flat = nnx.to_flat_state(nnx.state(b.actor, nnx.Param))
    for path, param in actor_flat:
        posterior_mean = b.posterior.layers[path].mean.get_value()
        assert jnp.allclose(posterior_mean, param.get_value().flatten(), atol=1e-6), (
            f"Posterior mean for '{path}' does not match actor param after sync"
        )


def test_frozen_inject_posterior_copies_mean_to_actor():
    """After inject, each actor param must equal the corresponding posterior mean (reshaped)."""
    gd, state = _make_bundle_and_split(init_std=0.5)  # distinct mean/std
    inject_fn = _make_frozen_inject_posterior(gd)
    new_state = inject_fn(state)

    b = nnx.merge(gd, new_state)
    actor_flat = nnx.to_flat_state(nnx.state(b.actor, nnx.Param))
    for path, param in actor_flat:
        posterior_mean = b.posterior.layers[path].mean.get_value()
        assert jnp.allclose(
            param.get_value(), posterior_mean.reshape(param.get_value().shape), atol=1e-6
        ), f"Actor param '{path}' does not equal posterior mean after inject"


def test_frozen_adaptive_continuous_update_changes_critics():
    gd, state = _make_bundle_and_split()
    action_scale = jnp.ones(ACT_DIM)
    action_bias  = jnp.zeros(ACT_DIM)
    update_fn = _make_frozen_adaptive_continuous_update(
        gd, action_scale, action_bias, adaptation_samples=2
    )

    obs      = jax.random.normal(jax.random.PRNGKey(0), (16, OBS_DIM))
    actions  = jax.random.normal(jax.random.PRNGKey(1), (16, ACT_DIM))
    rewards  = jax.random.uniform(jax.random.PRNGKey(2), (16, 1))
    next_obs = jax.random.normal(jax.random.PRNGKey(3), (16, OBS_DIM))
    dones    = jnp.zeros((16, 1))

    new_state, metrics = update_fn(
        state, obs, actions, rewards, next_obs, dones,
        gamma=0.99, key=jax.random.PRNGKey(4),
    )
    assert 'critic/loss' in metrics
    assert jnp.isfinite(metrics['critic/loss'])

    orig_leaves = jax.tree.leaves(state)
    new_leaves  = jax.tree.leaves(new_state)
    assert any(
        not jnp.array_equal(o, n) for o, n in zip(orig_leaves, new_leaves)
    ), "bundle_state unchanged after adaptive critic update"


def test_frozen_adaptive_discrete_update_changes_critics():
    obs_dim, n_actions = 4, 3
    actor   = MLP(obs_dim, n_actions, [16], rngs=nnx.Rngs(0))
    critic1 = MLP(obs_dim, n_actions, [16], rngs=nnx.Rngs(1))
    critic2 = MLP(obs_dim, n_actions, [16], rngs=nnx.Rngs(2))

    b = _PBSACBundle()
    b.actor        = actor
    b.actor_opt    = nnx.Optimizer(actor, optax.adam(1e-3), wrt=nnx.Param)
    b.critic1      = critic1
    b.critic1_opt  = nnx.Optimizer(critic1, optax.adam(1e-3), wrt=nnx.Param)
    b.critic2      = critic2
    b.critic2_opt  = nnx.Optimizer(critic2, optax.adam(1e-3), wrt=nnx.Param)
    b.target_critic1 = copy.deepcopy(critic1)
    b.target_critic2 = copy.deepcopy(critic2)
    b.log_alpha    = LogAlpha(0.0)
    b.posterior    = BlockPosterior.from_actor(actor, rank=2, init_std=0.01)
    b.prior        = BlockPrior.from_posterior(b.posterior)
    b.pb_optimizer = nnx.Optimizer(b.posterior, optax.adam(1e-3), wrt=nnx.Param)

    gd, state = nnx.split(b)
    update_fn = _make_frozen_adaptive_discrete_update(gd, adaptation_samples=2)

    B = 16
    obs      = jax.random.normal(jax.random.PRNGKey(0), (B, obs_dim))
    actions  = jax.random.randint(jax.random.PRNGKey(1), (B, 1), 0, n_actions)
    rewards  = jax.random.uniform(jax.random.PRNGKey(2), (B, 1))
    next_obs = jax.random.normal(jax.random.PRNGKey(3), (B, obs_dim))
    dones    = jnp.zeros((B, 1))

    new_state, metrics = update_fn(
        state, obs, actions, rewards, next_obs, dones,
        gamma=0.99, key=jax.random.PRNGKey(4),
    )
    assert 'critic/loss' in metrics
    assert any(
        not jnp.array_equal(o, n)
        for o, n in zip(jax.tree.leaves(state), jax.tree.leaves(new_state))
    )


# ── Mixing time estimation (numpy path, standard route) ──────────────────────

def _make_pbsac_agent():
    from kestrl.environments.registry import get_env_builder
    env = get_env_builder().build_env("Pendulum-v1", num_envs=1)
    return PBSAC(env=env, algo_cfg={'pac_bayes_active': True}, seed=42)


@pytest.fixture(scope="module")
def pbsac_agent():
    return _make_pbsac_agent()


def test_estimate_mixing_time_constant_signal(pbsac_agent):
    """Constant reward signal (var ≈ 0) → mixing time = 1."""
    traj = PBTrajectory(
        states=np.zeros((100, 3)), actions=np.zeros((100, 1)),
        rewards=np.ones(100), log_probs_b=np.zeros(100),
        mask=np.ones(100, dtype=bool), G=0.0,
    )
    pbsac_agent.mixing_time = 1
    mt = pbsac_agent._estimate_mixing_time([traj])
    assert mt == 1, f"Constant signal → mixing_time should be 1, got {mt}"


def test_estimate_mixing_time_slow_sine(pbsac_agent):
    """Slowly varying sine → autocorr stays high → mixing time > 1."""
    x = np.linspace(0, 4 * np.pi, 100)
    traj = PBTrajectory(
        states=np.zeros((100, 3)), actions=np.zeros((100, 1)),
        rewards=np.sin(x), log_probs_b=np.zeros(100),
        mask=np.ones(100, dtype=bool), G=0.0,
    )
    pbsac_agent.mixing_time = 1
    mt = pbsac_agent._estimate_mixing_time([traj])
    assert mt > 1, f"Slow sine → mixing_time should be > 1, got {mt}"


def test_estimate_mixing_time_white_noise_below_sine(pbsac_agent):
    """White noise decorrelates fast → smaller mixing time than slow sine."""
    x = np.linspace(0, 4 * np.pi, 100)
    sine_traj = PBTrajectory(
        states=np.zeros((100, 3)), actions=np.zeros((100, 1)),
        rewards=np.sin(x), log_probs_b=np.zeros(100),
        mask=np.ones(100, dtype=bool), G=0.0,
    )
    pbsac_agent.mixing_time = 1
    mt_sine = pbsac_agent._estimate_mixing_time([sine_traj])

    rng = np.random.default_rng(42)
    noise_traj = PBTrajectory(
        states=np.zeros((100, 3)), actions=np.zeros((100, 1)),
        rewards=rng.standard_normal(100), log_probs_b=np.zeros(100),
        mask=np.ones(100, dtype=bool), G=0.0,
    )
    pbsac_agent.mixing_time = 1
    mt_noise = pbsac_agent._estimate_mixing_time([noise_traj])

    assert mt_noise < mt_sine, (
        f"White noise ({mt_noise}) should have smaller mixing time than sine ({mt_sine})"
    )


# ── explore_prob decay ────────────────────────────────────────────────────────

def test_explore_prob_decay_formula():
    """Linear decay from init → final, flooring at final."""
    init, final, duration_frac = 0.5, 0.1, 0.5
    total, learning_starts = 1_000_000, 0

    def probe(step):
        return max(
            final,
            init - (step - learning_starts)
                   / max(1, total - learning_starts) * duration_frac,
        )

    assert np.isclose(probe(0),       init,  atol=1e-6), "Should start at init"
    assert probe(total)              <= final + 1e-6,     "Should reach (or pass) final"
    assert probe(total * 2)          == final,            "Should floor at final"

    # Monotonically non-increasing
    steps = range(0, total + 1, total // 20)
    vals  = [probe(s) for s in steps]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1], (
            f"explore_prob increased at step {list(steps)[i]}: "
            f"{vals[i]:.4f} → {vals[i+1]:.4f}"
        )


def test_explore_prob_logged_in_metrics():
    """train_step must include 'explore_prob' in its returned metrics dict."""
    from kestrl.environments.registry import get_env_builder
    env   = get_env_builder().build_env("Pendulum-v1", num_envs=1)
    agent = PBSAC(env=env, algo_cfg={
        'total_timesteps':    500,
        'learning_starts':    10,
        'buffer_size':        200,
        'batch_size':         16,
        'pb_update_freq':     999_999,
        'pb_reset_prior_freq': 999_999,
    }, seed=0)

    for _ in range(15):
        agent.collect_rollouts(1)

    metrics = agent.train_step()
    assert 'explore_prob' in metrics, "explore_prob must be logged in train_step metrics"
    assert 0.0 < metrics['explore_prob'] <= 1.0
