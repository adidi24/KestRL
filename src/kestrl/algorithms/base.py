"""Base algorithm class for KestRL.

Provides shared infrastructure: env setup, logging, evaluation, checkpointing.
Algorithm-specific logic (networks, losses, train_step) lives in subclasses.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from kestrl.utils import set_seed


class BaseAlgorithm(ABC):
    """Base class for all RL algorithms.
    
    Handles:
      - Environment info extraction (obs_dim, action_dim, is_discrete)
      - Global step / episode tracking
      - Evaluation loop
      - Logging (tensorboard/wandb)
      - Checkpointing (orbax)
    
    Subclasses implement:
      - _build_networks()  → create actor, critic, etc.
      - train_step()       → one gradient update
      - get_action()       → policy inference
      - save() / load()    → checkpoint
    """

    def __init__(self, env, algo_cfg: dict[str, Any], *, seed: int = 0):
        self.env = env
        self.config = algo_cfg
        self.key = set_seed(seed)
        
        # Writer
        self.writer = None

        # ── Environment info ──────────────────────────────────
        self.num_envs = env.num_envs
        self.obs_shape = env.single_observation_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))
        
        self.last_obs = None
        
        self.action_space = env.single_action_space
        if hasattr(self.action_space, 'n'):
            self.is_discrete = True
            self.action_dim = self.action_space.n
        else:
            self.is_discrete = False
            self.action_dim = int(np.prod(self.action_space.shape))

        # ── Training state ────────────────────────────────────
        self.global_step = 0
        self.episode_count = 0
        self.episode_returns: list[float] = []
        self.start_time = None


    @abstractmethod
    def _build_networks(self) -> None:
        """Create all neural networks (actor, critic, targets, optimizers)."""
        ...

    @abstractmethod
    def train_step(self) -> dict[str, float]:
        """Execute one training step. Returns metrics dict."""
        ...

    @abstractmethod
    def get_action(self, obs: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Select action(s) given observation(s)."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save algorithm state to path."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load algorithm state from path."""
        ...

    # ── Shared methods ────────────────────────────────────────

    def evaluate(
        self,
        eval_env=None,
        num_episodes: int = 100,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Evaluate current policy.
        
        Args:
            eval_env: Environment for evaluation (uses self.env if None)
            num_episodes: Number of episodes to run
            deterministic: Use deterministic actions
            seed: Random seed for eval env reset
            
        Returns:
            Dict with mean_return, std_return, mean_length, etc.
        """
        env = eval_env or self.env
        
        episode_returns = []
        episode_lengths = []
        
        obs, _ = env.reset(seed=seed)
        obs = jnp.array(obs, dtype=jnp.float32)

        while len(episode_returns) < num_episodes:
            action = self.get_action(obs, deterministic=deterministic)
            # Convert JAX array to numpy for env.step
            action_np = np.asarray(action)
            obs, reward, terminations, truncations, infos = env.step(action_np)
            obs = jnp.array(obs, dtype=jnp.float32)

            # Gymnasium autoreset: check for final_info
            if '_final_info' in infos and any(infos['_final_info']):                                                                                                                 
                final_info = infos['final_info']                                                                                                                                     
                if '_episode' in final_info:                                                                                                                                         
                    mask = final_info['_episode']                                                                                                                                    
                    ep = final_info['episode']                                                                                                                                       
                    for i in range(len(mask)):                                                                                                                                       
                        if mask[i]:                                                                                                                                                                                                                                                                       
                            episode_returns.append(float(ep['r'][i]))
                            episode_lengths.append(int(ep['l'][i]))

        returns = np.array(episode_returns[:num_episodes])
        lengths = np.array(episode_lengths[:num_episodes])

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'median_return': np.median(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_length': np.mean(lengths),
            'num_episodes': len(returns),
        }

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to tensorboard writer."""
        if self.writer is None:
            return
        step = step or self.global_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def get_sps(self) -> float:
        """Steps per second since training started."""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.global_step / max(elapsed, 1e-8)

    def print_progress(self) -> None:
        """Print one-line training status."""
        mean_return = np.mean(self.episode_returns[-100:]) if self.episode_returns else 0.0
        print(f"Step: {self.global_step:>8d} | "
              f"Episodes: {self.episode_count:>5d} | "
              f"SPS: {self.get_sps():>6.0f} | "
              f"Mean Return: {mean_return:>8.2f}")

    def _next_key(self) -> jax.Array:
        """Split and advance the PRNG key. Returns a fresh subkey."""
        self.key, subkey = jax.random.split(self.key)
        return subkey
