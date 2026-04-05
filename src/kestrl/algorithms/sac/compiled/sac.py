import copy
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from kestrl.algorithms.sac.sac import LogAlpha, _SACBundle
from kestrl.algorithms.sac.updates import (
    _make_frozen_soft_update,
    _make_frozen_continuous_get_action,
    _make_frozen_discrete_get_action,
)
from kestrl.algorithms.sac.compiled.rollout import _make_jax_scan_rollout
from kestrl.algorithms.sac.compiled.updates import (
    _make_gradient_scan_continuous,
    _make_gradient_scan_discrete,
)
from kestrl.buffers.jax_replay_buffer import JAXBufferState, _empty_buffer, _buffer_add
from kestrl.networks import MLP, MultiHeadMLP
from kestrl.environments.builders.brax_builder import BraxVectorEnv

# ── Carry pytree ──────────────────────────────────────────────────────────────

class CompiledSACCarry(NamedTuple):
    """Full training state carried across lax.scan steps.

    Under jax.vmap(train)(keys) every field has a leading num_seeds axis,
    giving each seed an entirely independent training run in the same dispatch.
    """
    bundle_state:   nnx.State       # network weights + optimizer states
    jax_env_state:  object
    buf:            JAXBufferState
    buf_pos:        jax.Array
    buf_full:       jax.Array
    global_step:    jax.Array
    rng:            jax.Array
    ep_return:      jax.Array       # (num_envs,) — accumulated return for current episode
    last_ep_return: jax.Array       # (num_envs,) — return of last completed episode


# ── Return type ───────────────────────────────────────────────────────────────

class CompiledSACFunctions(NamedTuple):
    """All callables returned by make_compiled_sac.

    train      : train(key) → (CompiledSACCarry, all_metrics)
                 Full training in one lax.scan. Compose with jax.vmap for seeds.
    init       : init(key) → CompiledSACCarry
                 Initialise carry only — used by CompiledTrainer's epoch loop.
    step_epoch : step_epoch(carry) → (CompiledSACCarry, epoch_metrics)
                 Run log_interval train steps. JIT'd. Used for live logging.
    evaluate   : evaluate(bundle_state, eval_env, num_episodes, seed) → dict
                 Numpy eval path — works on any carry.bundle_state after training.
    """
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
    is_discrete: bool,
    rngs: nnx.Rngs,
) -> _SACBundle:
    """Construct a full SAC bundle from an nnx.Rngs object.

    JAX-traceable: works inside jax.vmap when rngs holds a traced key, producing
    N independent weight initialisations without duplicating Python graph code.

    Discrete:   actor  obs → logits (act_dim)          plain MLP
                critic obs → Q(s,a) for all a           plain MLP, act_dim outputs
    Continuous: actor  obs → {mean, log_std}            MultiHeadMLP
                critic concat(obs, action) → scalar Q   plain MLP, 1 output
    """
    b = _SACBundle()

    if is_discrete:
        b.actor   = MLP(obs_dim, act_dim, hidden_dims, activation=activation, rngs=rngs)
        critic_in = obs_dim
        critic_out = act_dim
    else:
        b.actor   = MultiHeadMLP(
            obs_dim,
            head_configs={'mean': act_dim, 'log_std': act_dim},
            hidden_dims=hidden_dims,
            activation=activation,
            rngs=rngs,
        )
        critic_in  = obs_dim + act_dim
        critic_out = 1

    b.actor_opt = nnx.Optimizer(b.actor, optax.adam(lr_actor), wrt=nnx.Param)

    b.critic1 = MLP(critic_in, critic_out, hidden_dims, activation=activation, rngs=rngs)
    b.critic2 = MLP(critic_in, critic_out, hidden_dims, activation=activation, rngs=rngs)
    b.critic1_opt = nnx.Optimizer(b.critic1, optax.adam(lr_critic), wrt=nnx.Param)
    b.critic2_opt = nnx.Optimizer(b.critic2, optax.adam(lr_critic), wrt=nnx.Param)

    b.target_critic1 = copy.deepcopy(b.critic1)
    b.target_critic2 = copy.deepcopy(b.critic2)

    b.log_alpha = LogAlpha(0.0)
    if autotune_alpha:
        b.alpha_opt = nnx.Optimizer(b.log_alpha, optax.adam(lr_alpha), wrt=nnx.Param)

    return b

# ── Public factory ────────────────────────────────────────────────────────────

def make_compiled_sac(env: BraxVectorEnv, config: dict) -> CompiledSACFunctions:
    """Build pure JAX training functions for CompiledSAC.

    Args:
        env   : BraxVectorEnv.
        config: Algorithm hyperparameters.

    Returns:
        CompiledSACFunctions — see class docstring for the callable signatures.
    """
    # ── Parse config ──────────────────────────────────────────
    obs_dim = env.single_observation_space.shape[0]
    num_envs = env.num_envs

    action_space = env.single_action_space
    is_discrete = env.is_discrete
    act_dim = int(action_space.n) if is_discrete else action_space.shape[0]

    _actor_cfg = config.get('actor_network', {})
    hidden_dims = tuple(_actor_cfg.get('hidden_dims', config.get('hidden_dims', [256, 256])))
    activation = _actor_cfg.get('activation', config.get('activation', 'relu'))

    total_timesteps = config.get('total_timesteps', 1_000_000)
    train_freq = config.get('train_freq', 1)
    gradient_steps = config.get('gradient_steps', 1)
    batch_size = config.get('batch_size', 256)
    buffer_size = config.get('buffer_size', 25_000)
    learning_starts = config.get('learning_starts', 1024)
    gamma = config.get('gamma', 0.99)
    tau = float(config.get('tau', 0.005))
    lr_actor = float(config.get('lr_actor', 3e-4))
    lr_critic = float(config.get('lr_critic', 3e-4))
    lr_alpha = float(config.get('lr_alpha', 3e-4))
    autotune_alpha = config.get('autotune_alpha', True)
    log_interval = config.get('log_interval', 50)   # train steps per logging epoch
    target_update_interval = config.get('target_update_interval', 1)
    use_conditional_update = target_update_interval > 1   
                              
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
        action_bias = None

    # ── One-time Python setup ─────────────────────────────────
    def _build(rngs: nnx.Rngs) -> _SACBundle:
        return _make_bundle(
            obs_dim, act_dim, hidden_dims, activation,
            lr_actor, lr_critic, lr_alpha, autotune_alpha, is_discrete, rngs,
        )

    _canonical  = _build(nnx.Rngs(0))
    graphdef, _ = nnx.split(_canonical)
    del _canonical

    rollout_fn     = _make_jax_scan_rollout(
        graphdef, env.brax_env, is_discrete, action_scale, action_bias,
    )
    
    if is_discrete:
        grad_fn        = _make_gradient_scan_discrete(
            graphdef, batch_size, gradient_steps,
            target_entropy, gamma, autotune_alpha,
        )
        _, _get_action_det = _make_frozen_discrete_get_action(graphdef)
    else:
        grad_fn        = _make_gradient_scan_continuous(
            graphdef, batch_size, gradient_steps,
            action_scale, action_bias, target_entropy, gamma, autotune_alpha,
        )
        _, _get_action_det = _make_frozen_continuous_get_action(graphdef)
    
    soft_update_fn = _make_frozen_soft_update(graphdef)

    _empty_metrics: dict = {
        'critic/loss': jnp.float32(0.),
        'actor/loss':  jnp.float32(0.),
    }
    if autotune_alpha:
        _empty_metrics['alpha/loss']  = jnp.float32(0.)
        _empty_metrics['alpha/value'] = jnp.float32(0.)
    
    # ── Shared scan body ──────────────────────────────────────
    def _train_step(carry: CompiledSACCarry, _):
        rng, k_roll, k_grad = jax.random.split(carry.rng, 3)
        prng_keys = jax.random.split(k_roll, train_freq)
        
        # 1. Rollout
        new_jax_env_state, trajectory = rollout_fn(
            carry.bundle_state, carry.jax_env_state, prng_keys,
        )
        
        # 2. Buffer write
        new_buf, new_pos, new_full = _buffer_add(
            carry.buf, carry.buf_pos, carry.buf_full, trajectory, rows,
        )
        upper    = jnp.where(new_full, rows, new_pos)
        new_step = carry.global_step + jnp.int32(train_freq * num_envs)

        # 3. Episode return tracking (independent of gradient update)
        # trajectory.reward/done: (train_freq, num_envs)
        step_reward    = trajectory.reward.sum(axis=0)            # (num_envs,)
        any_done       = trajectory.done.any(axis=0)              # (num_envs,) bool
        new_ep         = carry.ep_return + step_reward            # (num_envs,)
        new_last_ep    = jnp.where(any_done, new_ep, carry.last_ep_return)
        next_ep        = jnp.where(any_done, jnp.zeros_like(new_ep), new_ep)
        
        # 4. Gradient update (gated on learning_starts)
        def do_update(bs):
            return grad_fn(bs, new_buf, k_grad, upper)

        def skip_update(bs):
            return bs, _empty_metrics
        
        new_bundle, metrics = jax.lax.cond(
            new_step >= learning_starts,
            do_update,
            skip_update,
            carry.bundle_state,
        )
        metrics = dict(metrics)
        metrics['episode/return'] = new_last_ep.mean()
        
        # 5. Soft update
        if use_conditional_update:
            new_bundle = jax.lax.cond(                                                                  
                new_step % target_update_interval == 0,
                lambda b: soft_update_fn(b, tau),
                lambda b: b,
                new_bundle,                                                                             
            )
        else:                                                                                           
            new_bundle = soft_update_fn(new_bundle, tau)
        
        new_carry = CompiledSACCarry(
            bundle_state = new_bundle,
            jax_env_state = new_jax_env_state,
            buf = new_buf,
            buf_pos = new_pos,
            buf_full = new_full,
            global_step = new_step,
            rng = rng,
            ep_return = next_ep,
            last_ep_return = new_last_ep,
        )
        return new_carry, metrics
    
    # ── init ─────────────────────────────────────────────────

    def init(key: jax.Array) -> CompiledSACCarry:
        """Initialise carry from a PRNGKey."""
        k_init, k_env = jax.random.split(key)
        fresh_bundle    = _build(nnx.Rngs(params=k_init))
        _, bundle_state = nnx.split(fresh_bundle)
        return CompiledSACCarry(
            bundle_state   = bundle_state,
            jax_env_state     = env.jax_reset(k_env),
            buf            = _empty_buffer(rows, num_envs, obs_dim, act_dim),
            buf_pos        = jnp.int32(0),
            buf_full       = jnp.bool_(False),
            global_step    = jnp.int32(0),
            rng            = key,
            ep_return      = jnp.zeros(num_envs, dtype=jnp.float32),
            last_ep_return = jnp.zeros(num_envs, dtype=jnp.float32),
        )
    
    # ── train (full scan) ────────────────────────────────────

    def train(key: jax.Array) -> tuple[CompiledSACCarry, dict]:
        """Full training in one lax.scan. Compose with jax.vmap for multi-seed.

        Returns:
            carry      : final CompiledSACCarry (one per seed under vmap).
            all_metrics: dict with leaves of shape (num_train_steps,)
                         — or (num_seeds, num_train_steps) under vmap.
        """
        return jax.lax.scan(
            _train_step, init(key), None, length=num_train_steps,
        )
    
    # ── step_epoch (epoch scan) ──────────────────────────────

    @jax.jit
    def step_epoch(carry: CompiledSACCarry) -> tuple[CompiledSACCarry, dict]:
        """Run log_interval train steps and return their metrics.

        Used by CompiledTrainer for live logging: call in a Python loop,
        log epoch_metrics between calls.

        Returns:
            new_carry   : updated CompiledSACCarry.
            epoch_metrics: dict with leaves of shape (log_interval,).
        """
        return jax.lax.scan(
            _train_step, carry, None, length=log_interval,
        )

    # ── evaluate ─────────────────────────────────────────────

    def evaluate(
        bundle_state,
        eval_env: BraxVectorEnv | None = None,
        num_episodes: int = 100,
        seed: int | None = None,
    ) -> dict:
        """Deterministic policy evaluation over the numpy (D2H) path."""
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

    return CompiledSACFunctions(
        train      = train,
        init       = init,
        step_epoch = step_epoch,
        evaluate   = evaluate,
    )