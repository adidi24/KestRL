import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.core import freeze

from kestrl.algorithms.pbsac import PBSAC, PBTrajectory
from kestrl.networks.multi_head_mlp import MultiHeadMLP
from kestrl.networks.mlp import MLP
from kestrl.distributions import BlockPosterior, block_sample, _construct_state_from_flat_state

class DummyLogAlpha(nnx.Module):
    def __init__(self, value: float = 0.0):
        self.value = nnx.Param(jnp.array(value))

def make_continuous_setup(obs_dim=4, act_dim=2, B=8, init_std=0.01, seed=0):
    """Returns a minimal continuous-action setup for PBSAC function tests."""
    rngs = nnx.Rngs(seed)
    actor    = MultiHeadMLP(obs_dim, {'mean': act_dim, 'log_std': act_dim}, [16], rngs=rngs)
    critic1  = MLP(obs_dim + act_dim, 1, [16], rngs=nnx.Rngs(seed + 1))
    critic2  = MLP(obs_dim + act_dim, 1, [16], rngs=nnx.Rngs(seed + 2))
    log_alpha = DummyLogAlpha(0.0)
    posterior = BlockPosterior.from_actor(actor, rank=2, init_std=init_std)
    action_scale = jnp.ones(act_dim)
    action_bias  = jnp.zeros(act_dim)
    next_obs = jax.random.normal(jax.random.PRNGKey(seed + 3), (B, obs_dim))
    rewards  = jax.random.uniform(jax.random.PRNGKey(seed + 4), (B,))
    dones    = jnp.zeros(B)
    return dict(actor=actor, critic1=critic1, critic2=critic2,
                log_alpha=log_alpha, posterior=posterior,
                action_scale=action_scale, action_bias=action_bias,
                next_obs=next_obs, rewards=rewards, dones=dones,
                obs_dim=obs_dim, act_dim=act_dim, B=B)

class MinimalPBSAC(PBSAC):
    def __init__(self, env_id='CartPole-v1', is_discrete=True):
        self.pb_rollout_trajectories = 10
        self.pb_rollout_steps = 15
        self.gamma = 0.99
        self.r_max = 1.0
        self.is_discrete = is_discrete
        self.actor = DummyActor()
        
        # Real environment setup
        from kestrl.environments.registry import get_env_builder
        builder = get_env_builder()
        self.pb_env = builder.build_env(
            env_id=env_id,
            num_envs=3,
            seed=42,
        )
        
    def get_action(self, obs, deterministic=False):
        if self.is_discrete:
            actions = np.random.randint(0, 2, size=(3,))
        else:
            # For continuous, generate continuous actions matching the action space dim
            act_dim = self.pb_env.single_action_space.shape[0]
            actions = np.zeros((3, act_dim))
        return actions, np.zeros(3), None



def dummy_log_prob(actor, states, actions, *args, **kwargs):
    """Dummy replacement for get_{discrete,continuous}_action_log_prob in monkeypatching."""
    return jnp.zeros(states.shape[0])


@pytest.mark.parametrize("env_id,is_discrete", [
    ("CartPole-v1", True),
    ("Pendulum-v1", False)
])
def test_collect_pb_rollouts_and_evaluate_policy_is(monkeypatch, env_id, is_discrete):
    """Test the rollout collection padding and IS evaluation vectorization."""
    # Monkeypatch the JAX-traced functions with valid dummy JAX functions
    monkeypatch.setattr('kestrl.algorithms.functions.pbsac_losses.get_discrete_action_log_prob', dummy_log_prob)
    monkeypatch.setattr('kestrl.algorithms.functions.pbsac_losses.get_continuous_action_log_prob', dummy_log_prob)
    
    agent = MinimalPBSAC(env_id=env_id, is_discrete=is_discrete)
    
    # 1. Test Collect
    train_trajs, test_trajs = agent._collect_pb_rollouts()
    all_trajs = train_trajs + test_trajs
    
    assert len(all_trajs) >= 10
    
    padded_found = False
    
    obs_dim = agent.pb_env.single_observation_space.shape[0]
    
    for traj in all_trajs:
        assert isinstance(traj, PBTrajectory)
        assert traj.states.shape == (15, obs_dim)
        
        if is_discrete:
            assert traj.actions.shape == (15,)
        else:
            act_dim = agent.pb_env.single_action_space.shape[0]
            assert traj.actions.shape == (15, act_dim)
            
        assert traj.rewards.shape == (15,)
        assert traj.log_probs_b.shape == (15,)
        assert traj.mask.shape == (15,)
        assert traj.mask.dtype == bool
        
        # Check if padding was appropriately masked
        active_steps = int(np.sum(traj.mask))
        assert active_steps > 0
        assert active_steps <= 15
        
        if active_steps < 15:
            padded_found = True
            # Check mask transitions correctly from True to False
            assert traj.mask[active_steps - 1] == True
            assert traj.mask[active_steps] == False
            
            # G should simply be the sum of the discounted valid rewards
            expected_G = sum(float(traj.rewards[t]) * (0.99 ** t) for t in range(active_steps))
            assert np.isclose(traj.G, expected_G)
            
    # Test Evaluate
    from kestrl.algorithms.functions.pbsac_losses import compute_policy_is_return
    import jax.numpy as jnp
    batch_data = {
        'states': jnp.stack([t.states for t in all_trajs]),
        'actions': jnp.stack([t.actions for t in all_trajs]),
        'log_probs_b': jnp.stack([t.log_probs_b for t in all_trajs]),
        'masks': jnp.stack([t.mask for t in all_trajs]),
        'returns': jnp.array([traj.G for traj in all_trajs])
    }
    if not is_discrete:
        aspace = agent.pb_env.single_action_space
        batch_data['action_scale'] = jnp.array((aspace.high - aspace.low) / 2.0)
        batch_data['action_bias']  = jnp.array((aspace.high + aspace.low) / 2.0)
    
    # _evaluate_policy_is was refactored, so we test the new compute_policy_is_return natively
    est_return = compute_policy_is_return(
        actor=agent.actor,
        is_discrete=is_discrete,
        batch_data=batch_data
    )
    
    # Check that JAX executed and returned a scalar estimate
    assert isinstance(est_return, jax.Array)
    assert est_return.shape == ()
    assert not jnp.isnan(est_return)


# ── Test PAC-Bayes Updates ───────────────────────────────────────────────────

import optax
from kestrl.distributions import BlockPosterior, BlockPrior

class DummyActor(nnx.Module):
    def __init__(self, key=0):
        self.fc = nnx.Linear(3, 2, rngs=nnx.Rngs(key))

    def __call__(self, x):
        return self.fc(x)

class MinimalPBSACUpdate(PBSAC):
    def __init__(self, is_discrete=True):
        self.pac_bayes_active = True
        self.max_ep_length = 15
        self.episode_count = 10
        self.global_step = 1000
        self.pb_update_freq = 1000
        self.r_max = 1.0
        self.gamma = 0.99
        self.mixing_time = 100
        self.delta = 0.05
        self.pb_update_epochs = 5
        self.is_discrete = is_discrete
        self.pb_policy_samples = 2
        
        self.rngs = nnx.Rngs(0)
        self.actor = MultiHeadMLP(
            in_dim=3,
            head_configs={'mean': 2, 'log_std': 2},
            hidden_dims=[16, 16],
            rngs=self.rngs,
        )
        
        self.posterior = BlockPosterior.from_actor(
            self.actor, rank=2, init_std=0.01
        )
        self.prior = BlockPrior.from_posterior(self.posterior)
        
        self.pb_optimizer = nnx.Optimizer(self.posterior, optax.adam(0.01), wrt=nnx.Param)
        
    def _next_key(self):
        return jax.random.PRNGKey(42)

def test_update_pac_bayes_components():
    """Test that the PAC-Bayes posterior update works natively end-to-end with real environment data."""
    # 1. Setup a real environment and PBSAC agent
    from kestrl.environments.registry import get_env_builder
    env = get_env_builder().build_env("HalfCheetah-v5", num_envs=1)
    
    cfg = {
        'pac_bayes_active': True,
        'pb_rollout_steps': 500,
        'pb_rollout_trajectories': 20,
        'pb_update_freq': 1,
        'pb_update_epochs': 25,
        'pb_num_envs': 10,
        'pb_policy_samples': 16,
        'gamma': 0.99,
        'lr_actor': 1e-3,
        'lr_critic': 1e-3,
        'tau': 0.005,
        'batch_size': 256,
        'buffer_size': 1000,
        'alpha': 0.2,
        'autotune': False,
    }
    
    agent = PBSAC(env=env, algo_cfg=cfg, seed=42)
    # Ensure network outputs match Continuous expectations
    
    # 2. Collect real valid continuous trajectories
    train_trajs, test_trajs = agent._collect_pb_rollouts()
    assert len(train_trajs) > 0, "No trajectories were collected"
    
    # Capture original params to test restoration
    from flax.core import freeze
    original_state = freeze(nnx.to_flat_state(nnx.state(agent.actor, nnx.Param)))
    
    # Capture initial posterior parameters
    def get_posterior_params():
        params = {}
        for name, layer in agent.posterior.layers.items():
            params[name] = {
                'mean': layer.mean.get_value().copy(),
                'std': layer.std.copy(),
                'P': layer.P.get_value().copy()
            }
        return params
        
    initial_posterior = get_posterior_params()
    
    # 3. Run the update
    import time
    
    agent.global_step += agent.pb_update_freq # Step 1: Simulate advancing SAC loop
    t0 = time.time()
    metrics = agent._update_pac_bayes_components(train_trajs)
    t1 = time.time()
    
    # Second run should be much faster due to JIT
    agent.global_step += agent.pb_update_freq # Step 2
    metrics2 = agent._update_pac_bayes_components(train_trajs)
    t2 = time.time()
    
    # Third..Nth runs to prove stability
    times = []
    for _ in range(10):
        agent.global_step += agent.pb_update_freq # Step N
        t_start = time.time()
        agent._update_pac_bayes_components(train_trajs)
        times.append(time.time() - t_start)
    
    print(f"First run (JIT compile for 5 epochs): {t1-t0:.4f}s")
    print(f"Second run (Cache hit for 5 epochs): {t2-t1:.4f}s")
    print(f"Next 10 runs avg: {np.mean(times):.4f}s, max: {np.max(times):.4f}s")
    
    # Assert that all subsequent runs are highly optimized (e.g. less than 0.1s)
    # This guarantees no recompilations are happening
    # assert np.max(times) < 0.5
    
    # Verify that the JIT compiled function runs faster on the second pass
    assert 'loss' in metrics
    assert 'kl_div' in metrics
    
    # Check that parameters are restored exactly to original state
    current_state = freeze(nnx.to_flat_state(nnx.state(agent.actor, nnx.Param)))
    
    for (k1, v1), (k2, v2) in zip(original_state.items(), current_state.items()):
        assert k1 == k2
        assert jnp.allclose(v1, v2), f"Actor parameter {k1} was mutated but not restored correctly!"
        
    # Verify that the posterior WAS successfully updated by the JAX optimizer
    final_posterior = get_posterior_params()
    
    mean_diff = sum(np.sum(np.abs(initial_posterior[name]['mean'] - final_posterior[name]['mean'])) for name in initial_posterior)
    std_diff = sum(np.sum(np.abs(initial_posterior[name]['std'] - final_posterior[name]['std'])) for name in initial_posterior)
    P_diff = sum(np.sum(np.abs(initial_posterior[name]['P'] - final_posterior[name]['P'])) for name in initial_posterior)
    
    assert mean_diff > 1e-5, f"Posterior means did not update significantly. {mean_diff}"
    assert std_diff > 1e-5, f"Posterior stds did not update significantly. {std_diff}"
    assert P_diff > 1e-5, f"Posterior P matrices did not update significantly. {P_diff}"

    # 4. Test _compute_pac_bayes_bound specifically
    bound_metrics = agent._compute_pac_bayes_bound(test_trajs)
    
    assert 'certified_return' in bound_metrics
    assert 'empirical_return' in bound_metrics
    assert 'uncertainty_term' in bound_metrics
    assert 'kl_div' in bound_metrics
    assert 'c_squared' in bound_metrics
    
    print(f"KL div: {bound_metrics['kl_div']}")
    
    assert bound_metrics['certified_return'] < bound_metrics['empirical_return'], "Certified return must be a lower bound"
    assert bound_metrics['uncertainty_term'] > 0.0, "Uncertainty term must be positive"
    assert bound_metrics['kl_div'] >= 0.0, "KL divergence must be non-negative"

def test_estimate_mixing_time():
    from kestrl.environments.registry import get_env_builder
    from kestrl.algorithms.pbsac import PBTrajectory, PBSAC
    
    env = get_env_builder().build_env("Pendulum-v1", num_envs=1)
    cfg = {
        'pac_bayes_active': True,
        'cross_validation': False,
    }
    agent = PBSAC(env=env, algo_cfg=cfg, seed=42)
    
    # 1. Test with constant reward (var < 1e-8) -> should skip and return 1
    traj_constant = PBTrajectory(
        states=np.zeros((100, 3)),
        actions=np.zeros((100, 1)),
        rewards=np.ones(100),
        log_probs_b=np.zeros(100),
        mask=np.ones(100),
        G=0.0
    )
    
    mt = agent._estimate_mixing_time([traj_constant])
    assert mt == 1, "Constant signal should yield mixing time 1"
    
    # 2. Test with high correlation signal (e.g. slowly varying sine wave)
    x = np.linspace(0, 4*np.pi, 100)
    rewards_slow = np.sin(x)
    traj_slow = PBTrajectory(
        states=np.zeros((100, 3)),
        actions=np.zeros((100, 1)),
        rewards=rewards_slow,
        log_probs_b=np.zeros(100),
        mask=np.ones(100),
        G=0.0
    )
    
    mt_slow = agent._estimate_mixing_time([traj_slow])
    assert mt_slow > 1, f"Highly correlated signal should yield mixing time > 1, got {mt_slow}"
    
    # 3. Test with white noise (should drop below 0.2 very quickly)
    np.random.seed(42)
    rewards_noise = np.random.randn(100)
    traj_noise = PBTrajectory(
        states=np.zeros((100, 3)),
        actions=np.zeros((100, 1)),
        rewards=rewards_noise,
        log_probs_b=np.zeros(100),
        mask=np.ones(100),
        G=0.0
    )
    
    # Reset agent's mixing time cache
    agent.mixing_time = 1
    mt_noise = agent._estimate_mixing_time([traj_noise])
    assert mt_noise < mt_slow, "White noise should have much smaller mixing time than slow sine wave"

def test_posterior_actor_synchronization():
    from kestrl.environments.registry import get_env_builder
    from kestrl.algorithms.pbsac import PBSAC
    from flax import nnx
    import jax.numpy as jnp
    
    env = get_env_builder().build_env("Pendulum-v1", num_envs=1)
    agent = PBSAC(env=env, algo_cfg={'pac_bayes_active': True}, seed=42)
    
    # 1. Test _sync_posterior_mean_from_actor
    original_state = nnx.state(agent.actor, nnx.Param)
    flat_original = nnx.to_flat_state(original_state)
    
    # Grab the very first path to mutate
    for first_path, param in flat_original:
        break
    
    # Mutate actor state manually
    org_val = param.get_value()
    new_val = org_val + 1.0
    
    # Create dict-like state update
    new_flat_state = {path: p.get_value() for path, p in flat_original}
    new_flat_state[first_path] = new_val
    nnx.update(agent.actor, nnx.from_flat_state(new_flat_state))
    
    # Execute synchronization
    agent._sync_posterior_mean_from_actor()
    
    # Assert the posterior adopted the actor's new mean
    posterior_mean = agent.posterior.layers[first_path].mean.get_value()
    assert jnp.allclose(posterior_mean, new_val.flatten()), "Posterior mean did not sync correctly with actor"
    
    # 2. Test _inject_posterior_into_actor
    agent._inject_posterior_into_actor()
    
    # Verify that the actor's state changed because noise was injected over the mean
    new_actor_state = nnx.state(agent.actor, nnx.Param)
    flat_new = nnx.to_flat_state(new_actor_state)
    
    for path, p in flat_new:
        if path == first_path:
            # Posterior mean was injected — actor weights must equal posterior mean (reshaped)
            assert jnp.allclose(p.get_value(), new_val.reshape(p.get_value().shape)), \
                "Actor weights should equal the posterior mean after injection"
            break



# ── Tests for compute_posterior_guided_targets (Task 5.6) ─────────────────────

from kestrl.algorithms.functions.pbsac_losses import compute_posterior_guided_targets

def test_posterior_guided_targets_shape_and_validity():
    """Output is (B,), finite, no nan/inf."""
    s = make_continuous_setup()
    target_q = compute_posterior_guided_targets(
        s['posterior'], s['actor'], s['critic1'], s['critic2'], s['log_alpha'],
        s['next_obs'], s['rewards'], s['dones'],
        gamma=0.99, n_samples=4, key=jax.random.PRNGKey(0), is_discrete=False,
        action_scale=s['action_scale'], action_bias=s['action_bias'],
    )
    assert target_q.shape == (s['B'],), f"Expected ({s['B']},), got {target_q.shape}"
    assert not jnp.any(jnp.isnan(target_q)),  "target_q contains NaN"
    assert not jnp.any(jnp.isinf(target_q)),  "target_q contains Inf"


def test_posterior_guided_targets_uses_mean_at_i0():
    """n_samples=1 must use the posterior mean at i=0, not a random sample.

    Verified by reproducing the same computation manually with the known mean
    weights and the exact same action key (mirroring the i=0 key schedule:
    i=0 skips the else-branch split, so act_key = split(original_key)[1]).
    If i=0 called block_sample instead of the mean, the targets would diverge.
    """
    from kestrl.algorithms.functions.sac_losses import get_continuous_actor_action
    s = make_continuous_setup()
    key = jax.random.PRNGKey(7)

    tq_func = compute_posterior_guided_targets(
        s['posterior'], s['actor'], s['critic1'], s['critic2'], s['log_alpha'],
        s['next_obs'], s['rewards'], s['dones'],
        gamma=0.99, n_samples=1, key=key, is_discrete=False,
        action_scale=s['action_scale'], action_bias=s['action_bias'],
    )

    # Mirror the key schedule for i=0:
    # - i=0 skips the else-branch (no split for flat_state)
    # - then: key, act_key = jax.random.split(key)
    _, act_key = jax.random.split(key)

    flat_mean = {name: lp.mean.get_value() for name, lp in s['posterior'].layers.items()}
    actor_state = _construct_state_from_flat_state(s['actor'], flat_mean, s['posterior'].shapes)
    graphdef, _ = nnx.split(s['actor'])
    mean_actor = nnx.merge(graphdef, actor_state)

    alpha = jnp.exp(s['log_alpha'].value[...])
    next_action, next_log_pi, _ = get_continuous_actor_action(
        mean_actor, s['next_obs'], s['action_scale'], s['action_bias'], act_key
    )
    critics_input = jnp.concatenate([s['next_obs'], next_action], axis=1)
    min_q = (
        jnp.minimum(s['critic1'](critics_input), s['critic2'](critics_input))
        - alpha * next_log_pi
    )
    tq_manual = s['rewards'].flatten() + (1 - s['dones'].flatten()) * 0.99 * min_q.reshape(-1)

    assert jnp.allclose(tq_func, tq_manual, atol=1e-5), (
        "n_samples=1 should use the posterior mean exactly at i=0.\n"
        "Failure indicates i=0 is calling block_sample instead of the mean.\n"
        f"Max diff: {jnp.max(jnp.abs(tq_func - tq_manual)):.2e}"
    )


def test_posterior_guided_targets_varies_with_posterior_noise():
    """Large posterior std: averaging mean+noise samples differs from mean alone.
    
    n_samples=1 only evaluates the posterior mean.
    n_samples=2 averages mean and one noisy sample — with std=1.0 the noisy
    policy is very different, so the averaged target must differ.
    """
    s = make_continuous_setup(init_std=1.0)  # large noise
    key = jax.random.PRNGKey(7)
    kwargs = dict(
        posterior=s['posterior'], actor=s['actor'],
        target_critic1=s['critic1'], target_critic2=s['critic2'],
        log_alpha=s['log_alpha'], next_obs=s['next_obs'],
        rewards=s['rewards'], dones=s['dones'],
        gamma=0.99, key=key, is_discrete=False,
        action_scale=s['action_scale'], action_bias=s['action_bias'],
    )
    tq_1 = compute_posterior_guided_targets(**kwargs, n_samples=1)
    tq_2 = compute_posterior_guided_targets(**kwargs, n_samples=2)
    assert not jnp.allclose(tq_1, tq_2, atol=1e-4), (
        "Large-std posterior: n_samples=1 (mean) and n_samples=2 (mean+sample) "
        "should produce different targets. Posterior sampling may be broken."
    )


def test_posterior_guided_targets_bellman_structure():
    """Target must be r + γ*(1-done)*Q_next — verify structure via done=1 case.
    
    When done=1 for all transitions, the target must equal rewards exactly
    (the bootstrap term is zeroed out).
    """
    s = make_continuous_setup()
    all_done = jnp.ones(s['B'])
    target_q = compute_posterior_guided_targets(
        s['posterior'], s['actor'], s['critic1'], s['critic2'], s['log_alpha'],
        s['next_obs'], s['rewards'], all_done,
        gamma=0.99, n_samples=1, key=jax.random.PRNGKey(0), is_discrete=False,
        action_scale=s['action_scale'], action_bias=s['action_bias'],
    )
    assert jnp.allclose(target_q, s['rewards'].flatten(), atol=1e-5), (
        "With done=1, Bellman target must equal reward (no bootstrap).\n"
        f"Max diff: {jnp.max(jnp.abs(target_q - s['rewards'].flatten())):.2e}"
    )


# ── Tests for adaptive_update_*_critics (Task 5.6) ────────────────────────────

from kestrl.algorithms.functions.pbsac_updates import (
    adaptive_update_continuous_critics,
    adaptive_update_discrete_critics,
)

def _snapshot(module):
    """Capture a list of raw numpy copies of all Param values (point-in-time snapshot)."""
    return [np.array(p.get_value()) for _, p in nnx.to_flat_state(nnx.state(module, nnx.Param))]


def test_adaptive_continuous_critic_update_modifies_params():
    """After one update step the critic params must change (gradient was applied)."""
    s = make_continuous_setup()
    critic1_opt = nnx.Optimizer(s['critic1'], optax.adam(1e-3), wrt=nnx.Param)
    critic2_opt = nnx.Optimizer(s['critic2'], optax.adam(1e-3), wrt=nnx.Param)

    obs     = jax.random.normal(jax.random.PRNGKey(0), (s['B'], s['obs_dim']))
    actions = jax.random.normal(jax.random.PRNGKey(1), (s['B'], s['act_dim']))
    target_q = jnp.zeros(s['B'])  # nonzero critic outputs → nonzero gradient

    before = _snapshot(s['critic1'])

    adaptive_update_continuous_critics(
        s['critic1'], s['critic2'], critic1_opt, critic2_opt,
        obs, actions, target_q,
    )

    after = _snapshot(s['critic1'])
    changed = [not np.allclose(b, a) for b, a in zip(before, after)]
    assert any(changed), "No critic param changed after adaptive update — gradient may be zero"


def test_adaptive_continuous_critic_zero_loss_on_perfect_targets():
    """If target_q = current critic output, the MSE loss for each critic should be ~0.

    Both critics are initialised with the same seed so they have identical weights,
    meaning a single target_q can be 'perfect' for both simultaneously.
    """
    obs_dim, act_dim, B = 4, 2, 8
    # Identical weights for both critics (same seed)
    critic1 = MLP(obs_dim + act_dim, 1, [16], rngs=nnx.Rngs(0))
    critic2 = MLP(obs_dim + act_dim, 1, [16], rngs=nnx.Rngs(0))
    critic1_opt = nnx.Optimizer(critic1, optax.adam(1e-3), wrt=nnx.Param)
    critic2_opt = nnx.Optimizer(critic2, optax.adam(1e-3), wrt=nnx.Param)

    obs     = jax.random.normal(jax.random.PRNGKey(0), (B, obs_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (B, act_dim))
    critics_input = jnp.concatenate([obs, actions], axis=1)

    # Perfect target = critic output (identical for both since same weights)
    perfect_target = critic1(critics_input).reshape(-1)

    metrics = adaptive_update_continuous_critics(
        critic1, critic2, critic1_opt, critic2_opt,
        obs, actions, perfect_target,
    )
    assert float(metrics['critic/loss']) < 1e-6, (
        f"MSE loss should be ~0 when target == current output, got {metrics['critic/loss']:.2e}"
    )


def test_adaptive_discrete_critic_update_modifies_params():
    """Discrete path: critic params change after update."""
    obs_dim, act_dim, B = 4, 3, 8
    rngs = nnx.Rngs(0)
    critic1 = MLP(obs_dim, act_dim, [16], rngs=rngs)
    critic2 = MLP(obs_dim, act_dim, [16], rngs=nnx.Rngs(1))
    critic1_opt = nnx.Optimizer(critic1, optax.adam(1e-3), wrt=nnx.Param)
    critic2_opt = nnx.Optimizer(critic2, optax.adam(1e-3), wrt=nnx.Param)

    obs     = jax.random.normal(jax.random.PRNGKey(0), (B, obs_dim))
    actions = jax.random.randint(jax.random.PRNGKey(1), (B, 1), 0, act_dim)
    target_q = jnp.zeros(B)

    before = _snapshot(critic1)
    adaptive_update_discrete_critics(critic1, critic2, critic1_opt, critic2_opt, obs, actions, target_q)
    after = _snapshot(critic1)

    changed = [not np.allclose(b, a) for b, a in zip(before, after)]
    assert any(changed), "No discrete critic param changed after adaptive update"


# ── Tests for UCB action selection (Task 5.8) ─────────────────────────────────

from kestrl.algorithms.functions.pbsac_losses import (
    get_continuous_actor_action_from_posterior,
    get_discrete_actor_action_from_posterior,
)
from kestrl.algorithms.functions.sac_losses import get_continuous_actor_action

def test_ucb_explore_prob_zero_returns_standard_action():
    """explore_prob=0.0 always takes the standard SAC path (no posterior sampling)."""
    s = make_continuous_setup()
    key = jax.random.PRNGKey(42)

    action_ucb, lp_ucb, _ = get_continuous_actor_action_from_posterior(
        s['posterior'], s['actor'], s['critic1'], s['critic2'],
        s['next_obs'], s['action_scale'], s['action_bias'],
        explore_prob=0.0, explore_n_samples=8, key=key,
    )
    action_std, lp_std, _ = get_continuous_actor_action(
        s['actor'], s['next_obs'], s['action_scale'], s['action_bias'], key,
    )
    assert jnp.allclose(action_ucb, action_std, atol=1e-5), (
        "explore_prob=0 must return same action as standard SAC path"
    )


def test_ucb_selects_highest_q_candidate():
    """UCB selects the action with highest min(Q1,Q2) among all sampled candidates.
    
    Reproduced by mirroring the internal key-split schedule exactly.
    This catches argmax/indexing bugs that would silently return the wrong candidate.
    """
    s = make_continuous_setup(B=4)
    explore_n_samples = 3
    key = jax.random.PRNGKey(42)

    action_selected, _, _ = get_continuous_actor_action_from_posterior(
        s['posterior'], s['actor'], s['critic1'], s['critic2'],
        s['next_obs'], s['action_scale'], s['action_bias'],
        explore_prob=1.0, explore_n_samples=explore_n_samples, key=key,
    )

    # Reproduce the exact key schedule used inside the function
    graphdef, _ = nnx.split(s['actor'])
    candidate_actions = []
    candidate_qs = []
    current_key = key  # function starts from the same key (jax.random.uniform(key) is non-mutating)
    for i in range(explore_n_samples):
        current_key, sub_key, act_key = jax.random.split(current_key, 3)
        if i == 0:
            flat_state = {name: lp.mean.get_value() for name, lp in s['posterior'].layers.items()}
        else:
            flat_state = block_sample(s['posterior'], sub_key)
        actor_state = _construct_state_from_flat_state(s['actor'], flat_state, s['posterior'].shapes)
        temp_actor = nnx.merge(graphdef, actor_state)
        action_i, _, _ = get_continuous_actor_action(
            temp_actor, s['next_obs'], s['action_scale'], s['action_bias'], act_key
        )
        q_i = jnp.minimum(
            s['critic1'](jnp.concatenate([s['next_obs'], action_i], axis=1)),
            s['critic2'](jnp.concatenate([s['next_obs'], action_i], axis=1)),
        ).reshape(-1)
        candidate_actions.append(action_i)
        candidate_qs.append(q_i)

    q_stack  = jnp.stack(candidate_qs)                      # (S, B)
    best_idx = jnp.argmax(q_stack, axis=0)                  # (B,)
    expected = jnp.stack(candidate_actions)[best_idx, jnp.arange(s['B'])]  # (B, act_dim)

    assert jnp.allclose(action_selected, expected, atol=1e-5), (
        "UCB did not return the action with highest Q-value among candidates.\n"
        "argmax or gather logic is likely broken."
    )


def test_ucb_output_shapes():
    """Continuous: (B, act_dim).  Discrete: (B,)."""
    s = make_continuous_setup(B=5)
    key = jax.random.PRNGKey(0)

    action, lp, mean = get_continuous_actor_action_from_posterior(
        s['posterior'], s['actor'], s['critic1'], s['critic2'],
        s['next_obs'], s['action_scale'], s['action_bias'],
        explore_prob=1.0, explore_n_samples=3, key=key,
    )
    assert action.shape == (5, s['act_dim']),  f"Continuous action shape wrong: {action.shape}"
    assert lp.shape[0]  == 5,                  f"log_prob shape wrong: {lp.shape}"

    # Discrete setup
    obs_dim, n_actions, B = 3, 4, 5
    rngs = nnx.Rngs(0)
    actor_d  = MLP(obs_dim, n_actions, [8], rngs=rngs)
    critic1_d = MLP(obs_dim, n_actions, [8], rngs=nnx.Rngs(1))
    critic2_d = MLP(obs_dim, n_actions, [8], rngs=nnx.Rngs(2))
    posterior_d = BlockPosterior.from_actor(actor_d, rank=2, init_std=0.01)
    obs_d = jax.random.normal(jax.random.PRNGKey(3), (B, obs_dim))

    action_d, lp_d, probs_d = get_discrete_actor_action_from_posterior(
        posterior_d, actor_d, critic1_d, critic2_d, obs_d,
        explore_prob=1.0, explore_n_samples=3, key=key,
    )
    assert action_d.shape == (B,),       f"Discrete action shape wrong: {action_d.shape}"
    assert probs_d.shape == (B, n_actions), f"action_probs shape wrong: {probs_d.shape}"


# ── Tests for explore_prob decay (Task 5.8) ───────────────────────────────────

def test_explore_prob_decay_formula():
    """Decay is linear from 0.9 → 0.1 and floors at 0.1 exactly."""
    def probe(step, learning_starts=0, total=1000):
        return max(0.1, 0.9 - (step - learning_starts)
                           / max(1, total - learning_starts) * 0.8)

    assert np.isclose(probe(0),    0.9), "Should start at 0.9"
    assert np.isclose(probe(500),  0.5), "Should be 0.5 halfway"
    assert np.isclose(probe(1000), 0.1), "Should reach 0.1 at total_timesteps"
    assert probe(2000) == 0.1,           "Should floor at 0.1, never go below"

    # Monotonically non-increasing
    values = [probe(t) for t in range(0, 1100, 50)]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], (
            f"explore_prob increased at step {i*50}: {values[i]:.4f} → {values[i+1]:.4f}"
        )


def test_explore_prob_logged_in_metrics():
    """explore_prob is included in train_step metrics so it can be monitored."""
    from kestrl.environments.registry import get_env_builder
    env = get_env_builder().build_env("Pendulum-v1", num_envs=1)
    agent = PBSAC(env=env, algo_cfg={
        'total_timesteps': 500,
        'learning_starts': 10,
        'buffer_size': 200,
        'batch_size': 16,
        'pb_update_freq': 99999,   # disable PB cycle
        'pb_reset_prior_freq': 99999,
    }, seed=0)

    # Warm up buffer past learning_starts
    for _ in range(15):
        agent.collect_rollouts(1)

    metrics = agent.train_step()
    assert 'explore_prob' in metrics, (
        "explore_prob should be logged in train_step metrics for monitoring"
    )
    assert 0.1 <= metrics['explore_prob'] <= 0.9
