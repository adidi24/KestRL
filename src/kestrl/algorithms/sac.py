"""Soft Actor-Critic (SAC) — Discrete + Continuous.

Supports both discrete and continuous action spaces with automatic entropy tuning.
Uses twin Q-networks and separate target networks for stability.

Architecture:
    Discrete:
        actor:   obs → logits (action_dim)     → softmax → pi(a|s)
        critic:  obs → Q(s, a) for all a       (action_dim outputs)
    
    Continuous:
        actor:   obs → {mean, log_std}         → tanh squashed Gaussian
        critic:  concat(obs, action) → Q(s,a)  (scalar output)
"""

import json
import time
import copy
from pathlib import Path
from typing import Any
from hydra.utils import instantiate

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from kestrl.algorithms.base import BaseAlgorithm
from kestrl.algorithms.functions.sac_losses import (
    get_continuous_actor_action,
    get_discrete_actor_action
)
from kestrl.algorithms.functions.sac_updates import (
    update_continuous_actor_alpha,
    update_continuous_critics,
    update_discrete,
)
from kestrl.networks import MLP, MultiHeadMLP
from kestrl.buffers.replay_buffer import ReplayBuffer
from kestrl.utils import soft_update


class LogAlpha(nnx.Module):
    """Tiny wrapper so log_alpha can be used with nnx.Optimizer."""
    def __init__(self, init_value: float = 0.0):
        self.value = nnx.Param(jnp.array(init_value))

class SAC(BaseAlgorithm):
    """Soft Actor-Critic with discrete bug fixes."""

    def __init__(self, env, algo_cfg: dict[str, Any], *, seed: int = 0):
        super().__init__(env, algo_cfg, seed=seed)

        # ── Hyperparameters ───────────────────────────────────
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.lr_actor = self.config.get('lr_actor', 2.5e-4)
        self.lr_critic = self.config.get('lr_critic', 1e-3)
        self.lr_alpha = self.config.get('lr_alpha', 1e-3)
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.batch_size = self.config.get('batch_size', 128)
        self.learning_starts = self.config.get('learning_starts', 10000)
        self.hidden_dims = tuple(self.config.get('hidden_dims', [64]))
        self.total_timesteps = self.config.get('total_timesteps', 500_000)
        self.train_freq = self.config.get('train_freq', 10)
        self.target_update_interval = self.config.get('target_update_interval', 500)
        self.policy_frequency = self.config.get('policy_frequency', 1)

        self.max_ep_length = 1
        self.episode_count = 1

        # ── Entropy tuning ────────────────────────────────────
        self.autotune_alpha = self.config.get('autotune_alpha', True)
        if self.is_discrete:
            # Target entropy: ratio * log(1/|A|) — negative, acts as entropy floor
            # When policy entropy < |target|, alpha increases to encourage exploration
            target_ratio = self.config.get('target_entropy_ratio', 0.25) # 0.98
            self.target_entropy = target_ratio * jnp.log(1.0 / self.action_dim)
        else:
            # Target entropy: -dim(A)
            self.target_entropy = -self.action_dim

        # ── Build everything ──────────────────────────────────
        self._build_networks()
        self._build_buffer()

    def _build_networks(self) -> None:
        """Create actor, critics, target critics, and optimizers."""
        rngs = nnx.Rngs(int(self._next_key()[0]))

        # ── Actor ─────────────────────────────────────────────
        actor_cfg = self.config.get('actor_network', None)
        if self.is_discrete:
            # Discrete: obs → logits for each action
            if actor_cfg is not None:
                actor_fn = instantiate(actor_cfg, _partial_=True)
                self.actor = actor_fn(self.obs_dim, self.action_dim, rngs=rngs)
            else:
                self.actor = MLP(
                    self.obs_dim, self.action_dim, [64], 
                    rngs=rngs,
                )
        else:
            # Continuous: obs → {mean, log_std}
            self.action_scale = jnp.array(
                    (self.action_space.high - self.action_space.low) / 2.0,
                    dtype=jnp.float32)
            self.action_bias = jnp.array(
                    (self.action_space.high + self.action_space.low) / 2.0,
                    dtype=jnp.float32)
            head_configs = {'mean': self.action_dim, 'log_std': self.action_dim}
            
            if actor_cfg is not None:
                actor_fn = instantiate(actor_cfg, _partial_=True)
                self.actor = actor_fn(self.obs_dim, head_configs=head_configs, rngs=rngs)
            else:
                self.actor = MultiHeadMLP(
                    self.obs_dim,
                    head_configs=head_configs,
                    hidden_dims=[64],
                    rngs=rngs,
                )

        # self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(self.lr_actor, eps=1e-4), wrt=nnx.Param)
        self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(self.lr_actor), wrt=nnx.Param)

        # ── Twin Critics ──────────────────────────────────────
        critic_cfg = self.config.get('critic_network', None)
        if self.is_discrete:
            # Discrete: obs → Q-value for each action
            critic_in = self.obs_dim
            critic_out = self.action_dim
        else:
            # Continuous: concat(obs, action) → scalar Q
            critic_in = self.obs_dim + self.action_dim
            critic_out = 1

        if critic_cfg is not None:
            critic_fn = instantiate(critic_cfg, _partial_=True)
            self.critic1 = critic_fn(critic_in, critic_out, rngs=rngs)
            self.critic2 = critic_fn(critic_in, critic_out, rngs=rngs)
        else:
            self.critic1 = MLP(critic_in, critic_out, [64], rngs=rngs)
            self.critic2 = MLP(critic_in, critic_out, [64], rngs=rngs)
        
        self.critic1_optimizer = nnx.Optimizer(
            self.critic1, optax.adam(self.lr_critic), wrt=nnx.Param,
        )
        self.critic2_optimizer = nnx.Optimizer(
            self.critic2, optax.adam(self.lr_critic), wrt=nnx.Param,
        )

        # ── Target Critics (frozen copies) ────────────────────
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # ── Alpha (entropy coefficient) ───────────────────────
        if self.autotune_alpha:
            self.log_alpha = LogAlpha(0.0)
            self.alpha_optimizer = nnx.Optimizer(
                self.log_alpha, optax.adam(self.lr_alpha), wrt=nnx.Param
            )
        else:
            self.log_alpha = LogAlpha(float(jnp.log(jnp.array(self.config.get('alpha', 0.2)))))


    def _build_buffer(self) -> None:
        """Create replay buffer."""
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            n_envs=self.num_envs,
        )

    # ── Action selection ──────────────────────────────────────

    def get_action(self, obs: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Select action given observation.
        
        Discrete: sample from softmax(logits)  or  argmax(logits)
        Continuous: sample from tanh(Normal(mean, std))  or  tanh(mean)
        """
        log_prob, action_probs = None, None
        
        key = self._next_key()
        if self.is_discrete:
            if deterministic:
                logits = self.actor(obs)
                return jnp.argmax(logits, axis=-1), log_prob, action_probs
            
            action, log_prob, action_probs = get_discrete_actor_action(self.actor, obs, key)
            return action, log_prob, action_probs
        
        else:
            if deterministic:
                logits = self.actor(obs)
                mean = jnp.tanh(logits['mean']) * self.action_scale + self.action_bias
                return mean, None, None

            action, log_prob, mean = get_continuous_actor_action(
                self.actor, obs, self.action_scale, self.action_bias, key)
            return action, log_prob, mean

    # ── Rollout collection ────────────────────────────────────

    def collect_rollouts(self, num_steps: int = 1) -> None:
        """Collect experience for replay buffer."""
        if self.last_obs is None:
            self.last_obs, _ = self.env.reset()
        
        episode_returns = []
        episode_lengths = []
        
        for _ in range(num_steps):
            self.global_step += self.env.num_envs
            
            self._obs_array = jnp.array(self.last_obs)
            
            if self.global_step < self.learning_starts:
                # Vectorized random action sampling
                if self.is_discrete:
                    actions = np.random.randint(0, self.action_space.n, size=(self.env.num_envs,))
                else:
                    actions = np.random.uniform(
                        self.action_space.low, self.action_space.high, 
                        size=(self.env.num_envs, self.action_space.shape[0])
                    )
            else:
                actions, _, _ = self.get_action(self._obs_array, deterministic=False)
                actions = np.asarray(actions)
            
            # Take environment step
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Record episode completions
            rollout_episodes = 0
            if '_final_info' in infos and any(infos['_final_info']):                                                                                                                 
                final_info = infos['final_info']                                                                                                                                     
                if '_episode' in final_info:                                                                                                                                         
                    mask = final_info['_episode']                                                                                                                                    
                    ep = final_info['episode']                                                                                                                                       
                    for i in range(len(mask)):                                                                                                                                       
                        if mask[i]:                                                                                                                                                  
                            rollout_episodes += 1                                                                                                                                    
                            self.episode_count += 1                                                                                                                                  
                            episode_returns.append(float(ep['r'][i]))
                            episode_lengths.append(int(ep['l'][i]))
            
            # Update max episode length
            if episode_lengths:
                self.max_ep_length = max(self.max_ep_length, max(episode_lengths))
            
            # Final observation handling
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for idx in range(self.num_envs):
                    if (terminations[idx] or truncations[idx]) and infos["final_observation"][idx] is not None:
                        real_next_obs[idx] = infos["final_observation"][idx]

            # Store transitions in replay buffer
            self.replay_buffer.add(
                obs=self.last_obs,
                next_obs=real_next_obs,
                action=actions,
                reward=rewards,
                done=terminations,
                infos=infos
            )
            
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            self.last_obs = next_obs
            
        
        # Return rollout metrics
        metrics = {}
        if episode_returns:
            metrics.update({
                'rollout/episodic_return': episode_returns,
                'rollout/episodic_length': episode_lengths,
                'rollout/episodes': rollout_episodes
            })
        
        return metrics

    # ── Training step ─────────────────────────────────────────

    def train_step(self):
        rollout_metrics = self.collect_rollouts(self.train_freq)
        if self.global_step < self.learning_starts:
            return rollout_metrics
        
        data = self.replay_buffer.sample(self.batch_size)
        key = self._next_key()
        
        # Always update critics
        if self.is_discrete:
            metrics = update_discrete(
                self.actor, self.critic1, self.critic2,
                self.target_critic1, self.target_critic2,
                self.actor_optimizer, self.critic1_optimizer, self.critic2_optimizer,
                self.log_alpha, self.alpha_optimizer,
                data.observations, data.actions, data.rewards,
                data.next_observations, data.dones,
                self.gamma, self.target_entropy, key)
        else:
            metrics = update_continuous_critics(
                self.actor, self.critic1, self.critic2,
                self.target_critic1, self.target_critic2,
                self.critic1_optimizer, self.critic2_optimizer,
                data.observations, data.actions, data.rewards,
                data.next_observations, data.dones,
                self.log_alpha, self.gamma,
                self.action_scale, self.action_bias, key)
            # Delayed actor + alpha update (TD3-style)
            if self.global_step % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    key = self._next_key()
                    actor_alpha_metrics = update_continuous_actor_alpha(
                            self.actor, self.critic1, self.critic2,
                            self.actor_optimizer, self.log_alpha, self.alpha_optimizer,
                            data.observations, self.action_scale, self.action_bias,
                            self.autotune_alpha, self.target_entropy, key)
                    metrics.update(actor_alpha_metrics)
        
        # Target network soft update
        if self.global_step % self.target_update_interval == 0:
            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)
        
        return {**rollout_metrics, **metrics}

    # ── Checkpointing ─────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save network params, optimizer states, and training metadata.

        Creates a directory at `path` containing:
          - arrays/        : orbax checkpoint (params + optimizer opt_states)
          - metadata.json  : training scalars (global_step, episode_count)

        The caller is responsible for using unique paths (e.g. step-stamped).
        """
        ckpt_dir = Path(path).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        arrays = {
            'actor':         nnx.state(self.actor),
            'critic1':       nnx.state(self.critic1),
            'critic2':       nnx.state(self.critic2),
            'target_critic1': nnx.state(self.target_critic1),
            'target_critic2': nnx.state(self.target_critic2),
            'log_alpha':     nnx.state(self.log_alpha),
            'actor_opt':     self.actor_optimizer.opt_state,
            'critic1_opt':   self.critic1_optimizer.opt_state,
            'critic2_opt':   self.critic2_optimizer.opt_state,
        }
        if self.autotune_alpha:
            arrays['alpha_opt'] = self.alpha_optimizer.opt_state

        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(ckpt_dir / 'arrays', arrays)
        checkpointer.wait_until_finished()

        with open(ckpt_dir / 'metadata.json', 'w') as f:
            json.dump({'global_step': self.global_step,
                       'episode_count': self.episode_count}, f)

    def load(self, path: str) -> None:
        """Restore from a checkpoint created by save().

        Requires networks to be already built (_build_networks runs in __init__),
        which provides the abstract structure needed by orbax restore.
        """
        ckpt_dir = Path(path).resolve()

        # Abstract target
        abstract = {
            'actor':         nnx.state(self.actor),
            'critic1':       nnx.state(self.critic1),
            'critic2':       nnx.state(self.critic2),
            'target_critic1': nnx.state(self.target_critic1),
            'target_critic2': nnx.state(self.target_critic2),
            'log_alpha':     nnx.state(self.log_alpha),
            'actor_opt':     self.actor_optimizer.opt_state,
            'critic1_opt':   self.critic1_optimizer.opt_state,
            'critic2_opt':   self.critic2_optimizer.opt_state,
        }
        if self.autotune_alpha:
            abstract['alpha_opt'] = self.alpha_optimizer.opt_state

        restored = ocp.StandardCheckpointer().restore(ckpt_dir / 'arrays',
                                                      target=abstract)

        nnx.update(self.actor,          restored['actor'])
        nnx.update(self.critic1,        restored['critic1'])
        nnx.update(self.critic2,        restored['critic2'])
        nnx.update(self.target_critic1, restored['target_critic1'])
        nnx.update(self.target_critic2, restored['target_critic2'])
        nnx.update(self.log_alpha,      restored['log_alpha'])

        self.actor_optimizer.opt_state   = restored['actor_opt']
        self.critic1_optimizer.opt_state = restored['critic1_opt']
        self.critic2_optimizer.opt_state = restored['critic2_opt']
        if self.autotune_alpha and 'alpha_opt' in restored:
            self.alpha_optimizer.opt_state = restored['alpha_opt']

        with open(ckpt_dir / 'metadata.json') as f:
            meta = json.load(f)
        self.global_step   = meta['global_step']
        self.episode_count = meta['episode_count']
