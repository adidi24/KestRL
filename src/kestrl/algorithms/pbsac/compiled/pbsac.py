import copy
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from kestrl.algorithms.sac.sac import LogAlpha
from kestrl.algorithms.pbsac.pbsac import _PBSACBundle
from kestrl.algorithms.sac.updates import (
    _make_frozen_soft_update,
    _make_frozen_continuous_get_action,
    _make_frozen_discrete_get_action
)
from kestrl.algorithms.pbsac.updates import (
    _make_frozen_sync_posterior,
    _make_frozen_prior_ema_update,
)
from kestrl.algorithms.pbsac.compiled.rollout import _make_jax_pb_rollout, _make_jax_scan_pge_rollout
from kestrl.algorithms.sac.compiled.updates import (
    _make_gradient_scan_discrete,
    _make_gradient_scan_continuous
)
from kestrl.algorithms.pbsac.compiled.updates import (
    _make_pb_posterior_scan,
    _make_adaptive_gradient_scan_continuous,
    _make_adaptive_gradient_scan_discrete
)
from kestrl.algorithms.pbsac.functions import (
    estimate_mixing_time,
    compute_pac_bayes_bound,
)
from kestrl.buffers.jax_replay_buffer import JAXBufferState, _empty_buffer, _buffer_add
from kestrl.networks import MLP, MultiHeadMLP
from kestrl.environments.builders.brax_builder import BraxVectorEnv

from kestrl.distributions import (
    BlockPosterior,
    BlockPrior,
)

# ── Carry pytree ──────────────────────────────────────────────────────────────

class CompiledPBSACCarry(NamedTuple):
    """Full training state carried across lax.scan steps.

    When vmapped, each state property gets a leading num_seeds axis for
    parallel independent training.

    Adaptation (critic recalibration after PAC-Bayes updates) is self-contained
    inside _do_pb_update — it does NOT need carry-level actor_frozen tracking.
    This keeps the main SAC scan body unconditionally cheap, avoiding the
    lax.cond→lax.select degradation under vmap that would otherwise force
    the expensive adaptive update to execute at every step.
    """
    bundle_state:       nnx.State
    jax_env_state:      object
    buf:                JAXBufferState
    buf_pos:            jax.Array
    buf_full:           jax.Array
    global_step:        jax.Array
    rng:                jax.Array
    ep_return:          jax.Array       # (num_envs,) — accumulated return for current episode
    last_ep_return:     jax.Array       # (num_envs,) — return of last completed episode
    r_max:              jax.Array       # float32 — running max |reward|
    mixing_time:        jax.Array       # int32 — Markov chain mixing time (non-decreasing)
    pb_episode_count:   jax.Array       # int32 — cumulative PB trajectory count (T in c²)
    # Last PB update metrics — stale between PB cycles, fresh when PB fires.
    pb_loss:            jax.Array
    pb_kl_div:          jax.Array
    pb_mean_return:     jax.Array
    pb_lambda:          jax.Array
    # Bound evaluation metrics
    pb_certified_return:  jax.Array
    pb_empirical_return:  jax.Array
    pb_uncertainty_term:  jax.Array


# ── Return type ───────────────────────────────────────────────────────────────

class CompiledPBSACFunctions(NamedTuple):
    train:               Callable
    init:                Callable
    step_epoch:          Callable
    vmap_step_epoch:     Callable   # Python-level PB trigger + vmapped SAC/PGE
    evaluate:            Callable


# ── Bundle factory ────────────────────────────────────────────────────────────

def _make_bundle(
    obs_dim: int,
    act_dim: int,
    hidden_dims: tuple,
    activation: str,
    lr_actor: float,
    lr_critic: float,
    lr_alpha: float,
    autotune_alpha: bool,
    pb_rank: int,
    pb_init_std: float,
    pb_posterior_lr: float,
    is_discrete: bool,
    rngs: nnx.Rngs,
    fixed_layers_depth: int = 0,
) -> _PBSACBundle:
    """Construct a full PB-SAC bundle from an nnx.Rngs object."""
    b = _PBSACBundle()

    if is_discrete:
        b.actor    = MLP(obs_dim, act_dim, hidden_dims, activation=activation, rngs=rngs)
        critic_in  = obs_dim
        critic_out = act_dim
    else:
        b.actor = MultiHeadMLP(
            obs_dim,
            head_configs={'mean': act_dim, 'log_std': act_dim},
            hidden_dims=hidden_dims,
            activation=activation,
            rngs=rngs,
        )
        critic_in  = obs_dim + act_dim
        critic_out = 1

    b.actor_opt   = nnx.Optimizer(b.actor, optax.adam(lr_actor), wrt=nnx.Param)
    b.critic1     = MLP(critic_in, critic_out, hidden_dims, activation=activation, rngs=rngs)
    b.critic2     = MLP(critic_in, critic_out, hidden_dims, activation=activation, rngs=rngs)
    b.critic1_opt = nnx.Optimizer(b.critic1, optax.adam(lr_critic), wrt=nnx.Param)
    b.critic2_opt = nnx.Optimizer(b.critic2, optax.adam(lr_critic), wrt=nnx.Param)

    b.target_critic1 = copy.deepcopy(b.critic1)
    b.target_critic2 = copy.deepcopy(b.critic2)

    b.log_alpha = LogAlpha(0.0)
    if autotune_alpha:
        b.alpha_opt = nnx.Optimizer(b.log_alpha, optax.adam(lr_alpha), wrt=nnx.Param)

    b.posterior    = BlockPosterior.from_actor(
        b.actor, rank=pb_rank, init_std=pb_init_std,
        fixed_layers_depth=fixed_layers_depth,
    )
    b.prior        = BlockPrior.from_posterior(b.posterior)
    b.pb_optimizer = nnx.Optimizer(b.posterior, optax.adam(pb_posterior_lr), wrt=nnx.Param)

    return b


# ── Public factory ────────────────────────────────────────────────────────────

def make_compiled_pbsac(env: BraxVectorEnv, config: dict) -> CompiledPBSACFunctions:
    """Build pure JAX training functions for CompiledPBSAC."""
    # ── Parse config ──────────────────────────────────────────
    obs_dim  = env.single_observation_space.shape[0]
    num_envs = env.num_envs

    action_space = env.single_action_space
    is_discrete  = env.is_discrete
    act_dim      = int(action_space.n) if is_discrete else action_space.shape[0]

    _actor_cfg  = config.get('actor_network', {})
    hidden_dims = tuple(_actor_cfg.get('hidden_dims', config.get('hidden_dims', [256, 256])))
    activation  = _actor_cfg.get('activation', config.get('activation', 'relu'))

    total_timesteps        = config.get('total_timesteps', 1_000_000)
    train_freq             = config.get('train_freq', 1)
    gradient_steps         = config.get('gradient_steps', 1)
    batch_size             = config.get('batch_size', 256)
    buffer_size            = config.get('buffer_size', 25_000)
    learning_starts        = config.get('learning_starts', 1024)
    gamma                  = config.get('gamma', 0.99)
    tau                    = float(config.get('tau', 0.005))
    lr_actor               = float(config.get('lr_actor', 3e-4))
    lr_critic              = float(config.get('lr_critic', 3e-4))
    lr_alpha               = float(config.get('lr_alpha', 3e-4))
    autotune_alpha         = config.get('autotune_alpha', True)
    log_interval           = config.get('log_interval', 50)
    target_update_interval = config.get('target_update_interval', 1)
    use_conditional_update = target_update_interval > 1

    # ── PB-SAC specific ────────────────────────────────────────
    pb_rank                 = config.get('pb_rank', 10)
    pb_init_std             = config.get('pb_init_std', 0.01)
    delta                   = config.get('delta', 0.1)
    pb_update_freq          = config.get('pb_update_freq', 20_000)
    pb_update_epochs        = config.get('pb_update_epochs', 100)
    pb_rollout_trajectories = config.get('pb_rollout_trajectories', 100)
    pb_rollout_steps        = config.get('pb_rollout_steps', 500)
    pb_policy_samples       = int(config.get('pb_policy_samples', 16))
    pb_posterior_lr         = config.get('pb_posterior_lr', 3e-4)
    pac_bayes_active        = config.get('pac_bayes_active', True)
    adaptation_samples      = config.get('adaptation_samples', 256)
    actor_freeze_steps      = config.get('actor_freeze_steps', 20)
    mixing_time_init        = config.get('mixing_time', 1)
    explore_prob_init       = config.get('explore_prob_init', 0.5)
    explore_prob_final      = config.get('explore_prob_final', 0.0)
    explore_prob_decay_duration = config.get('explore_prob_decay_duration', 0.5)
    explore_n_samples       = config.get('explore_n_samples', 8)
    pb_num_envs             = config.get('pb_num_envs', 10)
    pb_prior_decay          = float(config.get('pb_prior_decay', 0.99))
    env_id                  = config.get('env_id', None)
    fixed_layers_depth      = config.get('fixed_layers_depth', 0)

    # pb_update_freq must be >= one epoch's worth of env steps so the trigger
    # fires at most once per step_epoch call.
    steps_per_epoch = log_interval * train_freq * num_envs
    assert pb_update_freq >= steps_per_epoch, (
        f"pb_update_freq ({pb_update_freq}) must be >= steps_per_epoch ({steps_per_epoch}). "
        f"The PB trigger is checked once per epoch and can only fire once."
    )

    # ── Separate PB environment ─────────────────────────────────────────
    # The standard route creates a separate env for PB rollouts with
    # pb_num_envs (typically 10). We do the same here.
    import brax.envs as _brax_envs
    assert env_id is not None, (
        "env_id must be passed via config (set by run.py) to create the PB env."
    )
    _brax_name    = env_id.removeprefix('brax/')
    pb_brax_env   = _brax_envs.create(_brax_name, batch_size=pb_num_envs)

    # Train/test split for PB trajectories (90/10, matching standard route)
    _n_total_pb_trajs = (
        (pb_rollout_trajectories + pb_num_envs - 1) // pb_num_envs
    ) * pb_num_envs  # actual trajectories collected (rounded up to batch boundary)
    _n_train_pb = int(_n_total_pb_trajs * 0.9)
    _n_test_pb  = _n_total_pb_trajs - _n_train_pb
    assert _n_test_pb >= 1, (
        f"Need at least 1 test trajectory for bound evaluation, got "
        f"{_n_total_pb_trajs} total ({pb_rollout_trajectories} requested, "
        f"{pb_num_envs} pb_num_envs)"
    )

    # ── Entropy target ─────────────────────────────────────────
    if is_discrete:
        target_ratio   = config.get('target_entropy_ratio', 0.25)
        target_entropy = target_ratio * float(np.log(1.0 / act_dim))
    else:
        target_entropy = float(-act_dim)

    rows            = max(buffer_size // num_envs, 1)
    num_train_steps = total_timesteps // (train_freq * num_envs)

    if not is_discrete:
        action_scale = jnp.array(
            (env.single_action_space.high - env.single_action_space.low) / 2.0,
            dtype=jnp.float32,
        )
        action_bias = jnp.array(
            (env.single_action_space.high + env.single_action_space.low) / 2.0,
            dtype=jnp.float32,
        )
    else:
        action_scale = None
        action_bias  = None

    # ── One-time Python setup ─────────────────────────────────
    def _build(rngs: nnx.Rngs) -> _PBSACBundle:
        return _make_bundle(
            obs_dim, act_dim, hidden_dims, activation,
            lr_actor, lr_critic, lr_alpha, autotune_alpha,
            pb_rank, pb_init_std, pb_posterior_lr,
            is_discrete, rngs,
            fixed_layers_depth=fixed_layers_depth,
        )

    _canonical  = _build(nnx.Rngs(0))
    graphdef, _ = nnx.split(_canonical)
    del _canonical

    # ── Rollout + explore schedule ─────────
    rollout_fn = _make_jax_scan_pge_rollout(
        graphdef, env.brax_env, is_discrete, action_scale, action_bias, explore_n_samples,
    )
    _t = np.arange(num_train_steps, dtype=np.float32) * (train_freq * num_envs)
    _explore_probs = np.maximum(
        explore_prob_final,
        explore_prob_init + (explore_prob_final - explore_prob_init)
        * _t / max(explore_prob_decay_duration * total_timesteps, 1.0),
    )
    explore_schedule = jnp.array(
        np.random.default_rng(0).random(num_train_steps) < _explore_probs, dtype=jnp.bool_,
    )

    pb_rollout_fn = _make_jax_pb_rollout(
        graphdef, pb_brax_env, pb_num_envs,
        pb_rollout_trajectories, pb_rollout_steps, gamma,
        is_discrete, action_scale, action_bias,
    )

    if is_discrete:
        grad_fn          = _make_gradient_scan_discrete(
            graphdef, batch_size, gradient_steps,
            target_entropy, gamma, autotune_alpha,
        )
        adaptive_grad_fn = _make_adaptive_gradient_scan_discrete(
            graphdef, batch_size, gradient_steps,
            gamma, adaptation_samples,
        )
        _, _get_action_det = _make_frozen_discrete_get_action(graphdef)
    else:
        grad_fn          = _make_gradient_scan_continuous(
            graphdef, batch_size, gradient_steps,
            action_scale, action_bias, target_entropy, gamma, autotune_alpha,
        )
        adaptive_grad_fn = _make_adaptive_gradient_scan_continuous(
            graphdef, batch_size, gradient_steps,
            action_scale, action_bias, gamma, adaptation_samples,
        )
        _, _get_action_det = _make_frozen_continuous_get_action(graphdef)

    pb_grad_fn     = _make_pb_posterior_scan(
        graphdef, is_discrete, pb_policy_samples,
        pb_update_epochs, action_scale, action_bias,
    )
    soft_update_fn      = _make_frozen_soft_update(graphdef)
    sync_posterior_fn   = _make_frozen_sync_posterior(graphdef)
    prior_ema_fn        = _make_frozen_prior_ema_update(graphdef, pb_prior_decay)

    def _eval_bound(bundle_state, test_batch, T, r_max, mixing_time, key):
        """Evaluate the PAC-Bayes bound on held-out test trajectories."""
        b = nnx.merge(graphdef, bundle_state)
        batch = {
            'states':      test_batch.states,
            'actions':     test_batch.actions,
            'log_probs_b': test_batch.log_probs_b,
            'masks':       test_batch.masks,
            'returns':     test_batch.returns,
        }
        if not is_discrete:
            batch['action_scale'] = action_scale
            batch['action_bias']  = action_bias
        return compute_pac_bayes_bound(
            b.posterior, b.prior, b.actor,
            is_discrete, key, pb_policy_samples,
            T, pb_rollout_steps, r_max, mixing_time,
            gamma, delta, batch,
        )

    _empty_sac_metrics: dict = {'critic/loss': jnp.float32(0.), 'actor/loss': jnp.float32(0.)}
    if autotune_alpha:
        _empty_sac_metrics['alpha/loss']  = jnp.float32(0.)
        _empty_sac_metrics['alpha/value'] = jnp.float32(0.)

    # ── PB update — runs once per epoch when triggered, outside the SAC scan ──
    def _do_pb_update(carry: CompiledPBSACCarry, k_pb: jax.Array) -> CompiledPBSACCarry:
        """PB cycle: sync → rollout → posterior opt → bound eval → prior EMA → adapt.

        Matches the standard route’s train_step order (pbsac.py lines 546–574):
          1. Sync posterior mean ← current actor weights
          2. Collect PB rollouts with separate pb_env (pb_num_envs)
          3. Estimate mixing time
          4. Split trajectories 90/10 train/test
          5. Optimise posterior on train split (pb_grad_fn)
          6. Inject posterior mean → actor
          7. Evaluate bound on test split
          8. Prior EMA update  (μ₀ ← decay·μ_q + (1-decay)·μ₀)
          9. Adaptation: recalibrate critics to new posterior
        """
        k_rollout, k_update, k_adapt, k_bound = jax.random.split(k_pb, 4)

        # 1. Sync posterior mean ← actor (so posterior starts from current weights)
        synced_bundle = sync_posterior_fn(carry.bundle_state)

        # 2. PB rollouts with separate PB env + capture r_max
        traj_batch, pb_r_max = pb_rollout_fn(synced_bundle, k_rollout)
        new_r_max = jnp.maximum(carry.r_max, pb_r_max)

        # 3. Mixing time estimation
        tau_est         = estimate_mixing_time(traj_batch.rewards, traj_batch.states, traj_batch.masks)
        new_mixing_time = jnp.maximum(carry.mixing_time, tau_est)

        # 4. Train/test split (matching standard route’s 90/10)
        train_batch = jax.tree.map(lambda x: x[:_n_train_pb], traj_batch)
        test_batch  = jax.tree.map(lambda x: x[_n_train_pb:], traj_batch)

        # 5. Compute c² using PB trajectory count (not global_step)
        new_pb_episode_count = carry.pb_episode_count + jnp.int32(_n_total_pb_trajs)
        T = jnp.maximum(new_pb_episode_count.astype(jnp.float32), 1.0)
        c_squared = jnp.maximum(
            jnp.float32(1e-6),
            new_r_max ** 2
            * (1.0 - jnp.float32(gamma) ** (2 * pb_rollout_steps))
            / (T * (1.0 - jnp.float32(gamma) ** 2)),
        )
        C_const       = c_squared * new_mixing_time.astype(jnp.float32)
        C_prime_const = C_const * jnp.log(jnp.sqrt(jnp.float32(2.0)) / jnp.float32(delta))

        # 6. Optimise posterior on train split + inject → actor
        new_bundle, pb_metrics = pb_grad_fn(
            synced_bundle, C_const, C_prime_const, train_batch, k_update
        )

        # 7. Evaluate bound on held-out test split
        bound_metrics = _eval_bound(
            new_bundle, test_batch,
            T, new_r_max, new_mixing_time.astype(jnp.float32), k_bound,
        )

        # 8. Prior EMA update: μ₀ ← decay·μ_q + (1-decay)·μ₀
        new_bundle = prior_ema_fn(new_bundle)

        # 9. Adaptation: recalibrate critics to the new posterior
        upper = jnp.where(carry.buf_full, rows, carry.buf_pos)

        def _adapt_step(state_rng, _):
            bs, rng = state_rng
            rng, k = jax.random.split(rng)
            bs, _ = adaptive_grad_fn(bs, carry.buf, k, upper)
            bs = soft_update_fn(bs, tau)
            return (bs, rng), None

        (new_bundle, _), _ = jax.lax.scan(
            _adapt_step, (new_bundle, k_adapt), None, length=actor_freeze_steps
        )

        return carry._replace(
            bundle_state       = new_bundle,
            r_max              = new_r_max,
            mixing_time        = new_mixing_time,
            pb_episode_count   = new_pb_episode_count,
            pb_loss            = pb_metrics['loss'],
            pb_kl_div          = pb_metrics['kl_div'],
            pb_mean_return     = pb_metrics['mean_empirical_return'],
            pb_lambda          = pb_metrics['lambda'],
            pb_certified_return = bound_metrics['certified_return'],
            pb_empirical_return = bound_metrics['empirical_return'],
            pb_uncertainty_term = bound_metrics['uncertainty_term'],
        )

    # ── init ─────────────────────────────────────────────────

    def init(key: jax.Array) -> CompiledPBSACCarry:
        k_init, k_env  = jax.random.split(key)
        fresh_bundle   = _build(nnx.Rngs(params=k_init))
        _, bundle_state = nnx.split(fresh_bundle)
        return CompiledPBSACCarry(
            bundle_state        = bundle_state,
            jax_env_state       = env.jax_reset(k_env),
            buf                 = _empty_buffer(rows, num_envs, obs_dim, act_dim),
            buf_pos             = jnp.int32(0),
            buf_full            = jnp.bool_(False),
            global_step         = jnp.int32(0),
            rng                 = key,
            ep_return           = jnp.zeros(num_envs, dtype=jnp.float32),
            last_ep_return      = jnp.zeros(num_envs, dtype=jnp.float32),
            r_max               = jnp.float32(0.0),
            mixing_time         = jnp.int32(mixing_time_init),
            pb_episode_count    = jnp.int32(0),
            pb_loss             = jnp.float32(0.0),
            pb_kl_div           = jnp.float32(0.0),
            pb_mean_return      = jnp.float32(0.0),
            pb_lambda           = jnp.float32(0.0),
            pb_certified_return = jnp.float32(0.0),
            pb_empirical_return = jnp.float32(0.0),
            pb_uncertainty_term = jnp.float32(0.0),
        )

    # ── train (full scan, no PB — use step_epoch for PB training) ────────────

    def train(key: jax.Array) -> tuple[CompiledPBSACCarry, dict]:
        explore_flags = jnp.zeros(num_train_steps, dtype=jnp.bool_)
        return jax.lax.scan(_train_step, init(key), explore_flags, length=num_train_steps)

    # ── SAC/PGE scan body ──────────────────────────────────────────────────

    def _train_step(carry: CompiledPBSACCarry, should_explore: jax.Array):
        rng, k_roll, k_grad = jax.random.split(carry.rng, 3)
        prng_keys = jax.random.split(k_roll, train_freq)

        # 1. Rollout
        new_jax_env_state, trajectory = rollout_fn(
            carry.bundle_state, carry.jax_env_state, prng_keys, should_explore,
        )

        # 2. Buffer write
        new_buf, new_pos, new_full = _buffer_add(
            carry.buf, carry.buf_pos, carry.buf_full, trajectory, rows,
        )
        upper    = jnp.where(new_full, rows, new_pos)
        new_step = carry.global_step + jnp.int32(train_freq * num_envs)

        # 3. Episode tracking + r_max
        step_reward = trajectory.rewards.sum(axis=0)
        any_done    = trajectory.dones.any(axis=0)
        new_ep      = carry.ep_return + step_reward
        new_last_ep = jnp.where(any_done, new_ep, carry.last_ep_return)
        next_ep     = jnp.where(any_done, jnp.zeros_like(new_ep), new_ep)
        new_r_max   = jnp.maximum(carry.r_max, jnp.max(jnp.abs(trajectory.rewards)))

        # 4. SAC gradient update — always normal grad_fn, no adaptive branch.
        new_bundle, sac_metrics = jax.lax.cond(
            new_step >= learning_starts,
            lambda bs: grad_fn(bs, new_buf, k_grad, upper),
            lambda bs: (bs, _empty_sac_metrics),
            carry.bundle_state,
        )

        # 5. Soft target update
        if use_conditional_update:
            new_bundle = jax.lax.cond(
                new_step % target_update_interval == 0,
                lambda b: soft_update_fn(b, tau),
                lambda b: b,
                new_bundle,
            )
        else:
            new_bundle = soft_update_fn(new_bundle, tau)

        # 6. Metrics
        metrics = dict(sac_metrics)
        metrics['episode/return']            = new_last_ep.mean()
        metrics['pac_bayes/loss']            = carry.pb_loss
        metrics['pac_bayes/kl_div']          = carry.pb_kl_div
        metrics['pac_bayes/mean_return']     = carry.pb_mean_return
        metrics['pac_bayes/lambda']          = carry.pb_lambda
        metrics['pac_bayes/mixing_time']     = carry.mixing_time.astype(jnp.float32)
        metrics['pac_bayes/r_max']           = new_r_max
        metrics['pac_bayes/certified_return'] = carry.pb_certified_return
        metrics['pac_bayes/empirical_return'] = carry.pb_empirical_return
        metrics['pac_bayes/uncertainty_term'] = carry.pb_uncertainty_term

        new_carry = CompiledPBSACCarry(
            bundle_state        = new_bundle,
            jax_env_state       = new_jax_env_state,
            buf                 = new_buf,
            buf_pos             = new_pos,
            buf_full            = new_full,
            global_step         = new_step,
            rng                 = rng,
            ep_return           = next_ep,
            last_ep_return      = new_last_ep,
            r_max               = new_r_max,
            mixing_time         = carry.mixing_time,
            pb_episode_count    = carry.pb_episode_count,
            pb_loss             = carry.pb_loss,
            pb_kl_div           = carry.pb_kl_div,
            pb_mean_return      = carry.pb_mean_return,
            pb_lambda           = carry.pb_lambda,
            pb_certified_return = carry.pb_certified_return,
            pb_empirical_return = carry.pb_empirical_return,
            pb_uncertainty_term = carry.pb_uncertainty_term,
        )
        return new_carry, metrics

    @jax.jit
    def step_epoch(carry: CompiledPBSACCarry) -> tuple[CompiledPBSACCarry, dict]:
        rng, k_pb = jax.random.split(carry.rng)
        carry     = carry._replace(rng=rng)

        if pac_bayes_active:
            pb_trigger = (
                ((carry.global_step + jnp.int32(steps_per_epoch)) // jnp.int32(pb_update_freq))
                > (carry.global_step // jnp.int32(pb_update_freq))
            ) & (carry.global_step >= jnp.int32(learning_starts))

            carry = jax.lax.cond(
                pb_trigger,
                lambda c: _do_pb_update(c, k_pb),
                lambda c: c,
                carry,
            )

        step_start   = carry.global_step // jnp.int32(train_freq * num_envs)
        step_indices = jnp.clip(
            step_start + jnp.arange(log_interval, dtype=jnp.int32),
            jnp.int32(0), jnp.int32(num_train_steps - 1),
        )
        explore_flags = explore_schedule[step_indices]
        return jax.lax.scan(_train_step, carry, explore_flags, length=log_interval)

    # ── evaluate ─────────────────────────────────────────────

    def evaluate(
        bundle_state,
        eval_env: BraxVectorEnv | None = None,
        num_episodes: int = 100,
        seed: int | None = None,
    ) -> dict:
        _env = eval_env or env

        episode_returns: list[float] = []
        episode_lengths: list[int]   = []
        running_returns = np.zeros(_env.num_envs, dtype=np.float32)
        running_lengths = np.zeros(_env.num_envs, dtype=np.int32)

        obs_np, _ = _env.reset(seed=seed)
        obs = jnp.asarray(obs_np, dtype=jnp.float32)

        while len(episode_returns) < num_episodes:
            if is_discrete:
                action, _, _ = _get_action_det(bundle_state, obs)
            else:
                action, _, _ = _get_action_det(bundle_state, obs, action_scale, action_bias)
            obs_np, rewards, terminations, _, _ = _env.step(np.asarray(action))
            obs = jnp.asarray(obs_np, dtype=jnp.float32)

            running_returns += rewards
            running_lengths += 1

            for i in range(_env.num_envs):
                if terminations[i]:
                    episode_returns.append(float(running_returns[i]))
                    episode_lengths.append(int(running_lengths[i]))
                    running_returns[i] = 0.0
                    running_lengths[i] = 0

        returns = np.array(episode_returns[:num_episodes])
        lengths = np.array(episode_lengths[:num_episodes])
        return {
            'mean_return':   float(np.mean(returns)),
            'std_return':    float(np.std(returns)),
            'median_return': float(np.median(returns)),
            'min_return':    float(np.min(returns)),
            'max_return':    float(np.max(returns)),
            'mean_length':   float(np.mean(lengths)),
            'num_episodes':  len(returns),
        }

    # ── vmapped helpers — compiled once, reused every call ─────────────────

    _jit_vmap_pb_update = jax.jit(jax.vmap(
        lambda c, k: _do_pb_update(c, k),
    ))

    @jax.jit
    def _single_seed_epoch(carry: CompiledPBSACCarry):
        """Single-seed scan evaluated within vmap."""
        step_start   = carry.global_step // jnp.int32(train_freq * num_envs)
        step_indices = jnp.clip(
            step_start + jnp.arange(log_interval, dtype=jnp.int32),
            jnp.int32(0), jnp.int32(num_train_steps - 1),
        )
        explore_flags = explore_schedule[step_indices]
        return jax.lax.scan(_train_step, carry, explore_flags, length=log_interval)

    _vmapped_epoch = jax.jit(jax.vmap(_single_seed_epoch))

    # ── vmap_step_epoch ───────────────────────────────────────────────────────
    # Evaluates the PB trigger in pure Python. The PB update is only vmapped
    # when the trigger condition is met, avoiding lax.cond compilation overhead.

    def _should_pb_fire(step: int) -> bool:
        """Python-level PB trigger check."""
        if not pac_bayes_active or step < learning_starts:
            return False
        next_step = step + steps_per_epoch
        return (next_step // pb_update_freq) > (step // pb_update_freq)

    def vmap_step_epoch(all_carries: CompiledPBSACCarry) -> tuple[CompiledPBSACCarry, dict]:
        """Multi-seed JAX environment scan with Python-level PAC-Bayes trigger."""
        step = int(all_carries.global_step[0])

        if _should_pb_fire(step):
            # Split RNG for PB keys — one per seed
            splits = jax.vmap(jax.random.split)(all_carries.rng)
            new_rngs = splits[:, 0]
            k_pbs    = splits[:, 1]
            all_carries = all_carries._replace(rng=new_rngs)
            all_carries = _jit_vmap_pb_update(all_carries, k_pbs)

        return _vmapped_epoch(all_carries)

    return CompiledPBSACFunctions(
        train               = train,
        init                = init,
        step_epoch          = step_epoch,
        vmap_step_epoch     = vmap_step_epoch,
        evaluate            = evaluate,
    )
