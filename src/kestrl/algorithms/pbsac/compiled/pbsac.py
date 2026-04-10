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
from kestrl.algorithms.pbsac.compiled.rollout import _make_jax_scan_pge_rollout
from kestrl.algorithms.pbsac.compiled.rollout import _make_jax_pb_rollout
from kestrl.algorithms.sac.compiled.updates import (
    _make_gradient_scan_discrete,
    _make_gradient_scan_continuous
)
from kestrl.algorithms.pbsac.compiled.updates import (
    _make_pb_posterior_scan,
    _make_adaptive_gradient_scan_continuous,
    _make_adaptive_gradient_scan_discrete
)
from kestrl.buffers.jax_replay_buffer import JAXBufferState, _empty_buffer, _buffer_add
from kestrl.networks import MLP, MultiHeadMLP
from kestrl.environments.builders.brax_builder import BraxVectorEnv

from kestrl.distributions import (
    BlockPosterior,
    BlockPrior,
)
from kestrl.algorithms.pbsac.functions import estimate_mixing_time

# ── Carry pytree ──────────────────────────────────────────────────────────────

class CompiledPBSACCarry(NamedTuple):
    """Full training state carried across lax.scan steps.

    Under jax.vmap(train)(keys) every field has a leading num_seeds axis,
    giving each seed an entirely independent training run in the same dispatch.
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
    actor_frozen:       jax.Array       # bool scalar
    actor_freeze_until: jax.Array       # int32 scalar — step at which actor unfreezes
    r_max:              jax.Array       # float32 — running max |reward|, used for C_const
    mixing_time:        jax.Array       # int32 — Markov chain mixing time estimate (non-decreasing)
    explore_prob:       jax.Array
    # Last PB update metrics — stale between PB cycles, updated when PB fires.
    # Emitted every step so the trainer sees them at every log interval.
    pb_loss:            jax.Array
    pb_kl_div:          jax.Array
    pb_mean_return:     jax.Array
    pb_lambda:          jax.Array


# ── Return type ───────────────────────────────────────────────────────────────

class CompiledPBSACFunctions(NamedTuple):
    train:      Callable
    init:       Callable
    step_epoch: Callable
    evaluate:   Callable


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

    b.posterior   = BlockPosterior.from_actor(b.actor, rank=pb_rank, init_std=pb_init_std)
    b.prior       = BlockPrior.from_posterior(b.posterior)
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
    mixing_time             = config.get('mixing_time', 1)
    explore_prob_init       = config.get('explore_prob_init', 0.5)
    explore_prob_final      = config.get('explore_prob_final', 0.1)
    explore_prob_decay_duration = config.get('explore_prob_decay_duration', 0.5)
    explore_n_samples       = config.get('explore_n_samples', 8)

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
        )

    _canonical  = _build(nnx.Rngs(0))
    graphdef, _ = nnx.split(_canonical)
    del _canonical

    rollout_fn = _make_jax_scan_pge_rollout(
        graphdef, env.brax_env, is_discrete, action_scale, action_bias, explore_n_samples
    )

    pb_rollout_fn = _make_jax_pb_rollout(
        graphdef, env.brax_env, num_envs,
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
    soft_update_fn = _make_frozen_soft_update(graphdef)

    _empty_sac_metrics: dict = {'critic/loss': jnp.float32(0.), 'actor/loss': jnp.float32(0.)}
    if autotune_alpha:
        _empty_sac_metrics['alpha/loss']  = jnp.float32(0.)
        _empty_sac_metrics['alpha/value'] = jnp.float32(0.)

    # ── Shared scan body ──────────────────────────────────────
    def _train_step(carry: CompiledPBSACCarry, _):
        rng, k_roll, k_grad, k_pb, k_pge_explore = jax.random.split(carry.rng, 5)
        prng_keys = jax.random.split(k_roll, train_freq)

        # 1. Rollout
        # Linear decay schedule for posterior guided exploration probability
        explore_prob = jnp.maximum(
            jnp.float32(explore_prob_final),
            (jnp.float32(explore_prob_final) - jnp.float32(explore_prob_init))
            / jnp.float32(explore_prob_decay_duration * total_timesteps)
            * carry.global_step.astype(jnp.float32)
            + jnp.float32(explore_prob_init),
        )
        should_explore = jax.random.uniform(k_pge_explore) < explore_prob
        new_jax_env_state, trajectory = rollout_fn(
            carry.bundle_state, carry.jax_env_state, prng_keys, should_explore
        )

        # 2. Buffer write
        new_buf, new_pos, new_full = _buffer_add(
            carry.buf, carry.buf_pos, carry.buf_full, trajectory, rows,
        )
        upper    = jnp.where(new_full, rows, new_pos)
        new_step = carry.global_step + jnp.int32(train_freq * num_envs)

        # 3. Episode tracking + r_max update from training trajectory
        step_reward = trajectory.rewards.sum(axis=0)
        any_done    = trajectory.dones.any(axis=0)
        new_ep      = carry.ep_return + step_reward
        new_last_ep = jnp.where(any_done, new_ep, carry.last_ep_return)
        next_ep     = jnp.where(any_done, jnp.zeros_like(new_ep), new_ep)
        new_r_max   = jnp.maximum(carry.r_max, jnp.max(jnp.abs(trajectory.rewards)))

        # 4. SAC gradient update — actor_frozen selects adaptive (critics only) vs full update.
        #    adaptive_grad_fn only returns {'critic/loss'}; pad with zeros so both branches
        #    of lax.cond have identical pytree structure.
        def do_sac_update(bs):
            def frozen_update(b):
                new_b, m = adaptive_grad_fn(b, new_buf, k_grad, upper)
                return new_b, {**_empty_sac_metrics, **m}
            return jax.lax.cond(
                carry.actor_frozen,
                frozen_update,
                lambda b: grad_fn(b, new_buf, k_grad, upper),
                bs,
            )

        new_bundle, sac_metrics = jax.lax.cond(
            new_step >= learning_starts,
            do_sac_update,
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

        # 6. PB update — fires every pb_update_freq steps once learning has started.
        #    The inner pb_grad_fn runs pb_update_epochs gradient steps in its own lax.scan
        #    and returns average metrics across those epochs.
        #    Both branches of lax.cond carry forward the PB metrics: fresh when PB fires,
        #    stale otherwise. Emitting them every step means the trainer sees them at every
        #    log interval without any special casing.
        if pac_bayes_active:
            pb_trigger = (
                (new_step // jnp.int32(pb_update_freq)) > (carry.global_step // jnp.int32(pb_update_freq))
            ) & (new_step >= jnp.int32(learning_starts))

            def _do_pb(bs):
                k_rollout, k_update = jax.random.split(k_pb)
                traj_batch, _ = pb_rollout_fn(bs, k_rollout)

                # Estimate mixing time from fresh trajectories; keep it non-decreasing
                tau_est         = estimate_mixing_time(traj_batch.rewards, traj_batch.states, traj_batch.masks)
                new_mixing_time = jnp.maximum(carry.mixing_time, tau_est)

                # C_const depends on mixing_time so it lives here, not in the outer scope
                episode_count = jnp.maximum(new_step.astype(jnp.float32), 1.0)
                c_squared = jnp.maximum(
                    jnp.float32(1e-6),
                    new_r_max ** 2
                    * (1.0 - jnp.float32(gamma) ** (2 * pb_rollout_steps))
                    / (episode_count * (1.0 - jnp.float32(gamma) ** 2)),
                )
                C_const       = c_squared * new_mixing_time.astype(jnp.float32)
                C_prime_const = C_const * jnp.log(jnp.sqrt(jnp.float32(2.0)) / jnp.float32(delta))

                new_bs, pb_metrics = pb_grad_fn(bs, C_const, C_prime_const, traj_batch, k_update)
                return new_bs, pb_metrics, new_mixing_time

            def _skip_pb(bs):
                return bs, {
                    'loss':                  carry.pb_loss,
                    'kl_div':                carry.pb_kl_div,
                    'mean_empirical_return': carry.pb_mean_return,
                    'lambda':                carry.pb_lambda,
                }, carry.mixing_time

            new_bundle, pb_metrics, new_mixing_time = jax.lax.cond(
                pb_trigger, _do_pb, _skip_pb, new_bundle,
            )

            # Actor freeze: set when PB fires, cleared when global_step reaches the deadline
            new_actor_freeze_until = jax.lax.cond(
                pb_trigger,
                lambda _: new_step + jnp.int32(actor_freeze_steps),
                lambda _: carry.actor_freeze_until,
                None,
            )
            new_actor_frozen = new_step < new_actor_freeze_until
        else:
            pb_metrics = {
                'loss':                  carry.pb_loss,
                'kl_div':                carry.pb_kl_div,
                'mean_empirical_return': carry.pb_mean_return,
                'lambda':                carry.pb_lambda,
            }
            new_mixing_time        = carry.mixing_time
            new_actor_freeze_until = carry.actor_freeze_until
            new_actor_frozen       = jnp.bool_(False)

        # 7. Assemble step metrics
        metrics = dict(sac_metrics)
        metrics['episode/return']         = new_last_ep.mean()
        metrics['pac_bayes/loss']         = pb_metrics['loss']
        metrics['pac_bayes/kl_div']       = pb_metrics['kl_div']
        metrics['pac_bayes/mean_return']  = pb_metrics['mean_empirical_return']
        metrics['pac_bayes/lambda']       = pb_metrics['lambda']
        metrics['pac_bayes/mixing_time']  = new_mixing_time.astype(jnp.float32)
        metrics['pac_bayes/actor_frozen'] = new_actor_frozen.astype(jnp.float32)
        metrics['pac_bayes/r_max']        = new_r_max

        new_carry = CompiledPBSACCarry(
            bundle_state       = new_bundle,
            jax_env_state      = new_jax_env_state,
            buf                = new_buf,
            buf_pos            = new_pos,
            buf_full           = new_full,
            global_step        = new_step,
            rng                = rng,
            ep_return          = next_ep,
            last_ep_return     = new_last_ep,
            actor_frozen       = new_actor_frozen,
            actor_freeze_until = new_actor_freeze_until,
            r_max              = new_r_max,
            mixing_time        = new_mixing_time,
            explore_prob       = explore_prob,
            pb_loss            = pb_metrics['loss'],
            pb_kl_div          = pb_metrics['kl_div'],
            pb_mean_return     = pb_metrics['mean_empirical_return'],
            pb_lambda          = pb_metrics['lambda'],
        )
        return new_carry, metrics

    # ── init ─────────────────────────────────────────────────

    def init(key: jax.Array) -> CompiledPBSACCarry:
        k_init, k_env  = jax.random.split(key)
        fresh_bundle   = _build(nnx.Rngs(params=k_init))
        _, bundle_state = nnx.split(fresh_bundle)
        return CompiledPBSACCarry(
            bundle_state       = bundle_state,
            jax_env_state      = env.jax_reset(k_env),
            buf                = _empty_buffer(rows, num_envs, obs_dim, act_dim),
            buf_pos            = jnp.int32(0),
            buf_full           = jnp.bool_(False),
            global_step        = jnp.int32(0),
            rng                = key,
            ep_return          = jnp.zeros(num_envs, dtype=jnp.float32),
            last_ep_return     = jnp.zeros(num_envs, dtype=jnp.float32),
            actor_frozen       = jnp.bool_(False),
            actor_freeze_until = jnp.int32(0),
            r_max              = jnp.float32(0.0),
            mixing_time        = jnp.int32(mixing_time),   # seed from config, updated in-place
            explore_prob       = jnp.float32(explore_prob_init),
            pb_loss            = jnp.float32(0.0),
            pb_kl_div          = jnp.float32(0.0),
            pb_mean_return     = jnp.float32(0.0),
            pb_lambda          = jnp.float32(0.0),
        )

    # ── train (full scan) ────────────────────────────────────

    def train(key: jax.Array) -> tuple[CompiledPBSACCarry, dict]:
        return jax.lax.scan(_train_step, init(key), None, length=num_train_steps)

    # ── step_epoch (epoch scan) ──────────────────────────────

    @jax.jit
    def step_epoch(carry: CompiledPBSACCarry) -> tuple[CompiledPBSACCarry, dict]:
        """Run log_interval train steps and return their metrics.

        PB metrics in the returned dict have shape (log_interval,) like all other
        metrics. They hold the carry value at each step — stale between PB cycles,
        updated at the step the PB trigger fires. The trainer aggregates them the
        same way as SAC metrics; no special casing needed.
        """
        return jax.lax.scan(_train_step, carry, None, length=log_interval)

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

    return CompiledPBSACFunctions(
        train      = train,
        init       = init,
        step_epoch = step_epoch,
        evaluate   = evaluate,
    )
