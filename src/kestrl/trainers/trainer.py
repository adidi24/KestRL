"""Training controller for BenchRL-JAX.

Orchestrates training loop, logging, evaluation, and checkpointing.
Adapted from benchrl.trainers.trainer for JAX.
"""

import os
import shutil
import time
from collections import deque
from typing import Any

import numpy as np
import tqdm

from kestrl.algorithms.base import BaseAlgorithm


class Trainer:
    """Main training controller."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        config: dict[str, Any] | None = None,
        writer=None,
        eval_env=None,
    ):
        self.algorithm = algorithm
        self.config = config or {}
        self.writer = writer
        self.eval_env = eval_env

        # Training config
        self.total_timesteps = self.config.get('total_timesteps',
                                                algorithm.total_timesteps)
        self.eval_interval = self.config.get('eval_interval', None)
        self.eval_episodes = self.config.get('eval_episodes', 10)
        self.eval_deterministic = self.config.get('eval_deterministic', True)

        # Checkpointing
        self.checkpoint_dir = self.config.get('checkpoint_dir', None)
        self.checkpoint_interval = self.config.get('checkpoint_interval', None)
        self.max_checkpoints = self.config.get('max_checkpoints', None)
        self._checkpoint_paths: deque = deque()
        self._last_checkpoint_step: int = 0

        # Training state
        self.start_time: float | None = None
        self._last_return: float = 0.0

    # ── Main loop ─────────────────────────────────────────────

    def train(self) -> None:
        """Run full training loop."""
        self.start_time = time.time()
        self.algorithm.start_time = self.start_time

        print(f"Starting training for {self.total_timesteps} timesteps")
        print(f"Algorithm: {self.algorithm.__class__.__name__}")
        print(f"Backend: JAX")
        print("-" * 50)

        with tqdm.tqdm(total=self.total_timesteps, desc="Training") as pbar:
            while self.algorithm.global_step < self.total_timesteps:
                # One training step (collect + update)
                metrics = self.algorithm.train_step()

                # Log metrics
                self._log_metrics(metrics)

                # Update progress bar
                pbar.update(self.algorithm.global_step - pbar.n)

                if metrics.get('rollout/episodes', 0) != 0:
                    ep_returns = metrics.get('rollout/episodic_return', [])
                    if ep_returns:
                        self._last_return = float(np.mean(ep_returns))

                pbar.set_postfix({
                    'SPS': f"{self.algorithm.get_sps():.0f}",
                    'Return': f"{self._last_return:.2f}",
                    'critic_loss': f"{metrics.get('critic/loss', 0):.4f}",
                })

                # Periodic evaluation
                if (self.eval_interval is not None
                    and self.algorithm.global_step > 0
                    and self.algorithm.global_step % self.eval_interval == 0):
                    self._evaluate_policy()

                # Periodic checkpointing
                if (self.checkpoint_interval is not None
                        and self.checkpoint_dir is not None
                        and self.algorithm.global_step > 0
                        and self.algorithm.global_step - self._last_checkpoint_step
                            >= self.checkpoint_interval):
                    self._save_checkpoint()

        # Final evaluation
        if self.eval_env is not None:
            print("\nRunning final evaluation...")
            self._evaluate_policy()

        self._print_summary()

        if self.writer is not None:
            self.writer.close()

    # ── Logging ───────────────────────────────────────────────

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        if self.writer is None:
            return

        step = self.algorithm.global_step
        for key, value in metrics.items():
            if isinstance(value, list):
                for v in value:
                    self.writer.add_scalar(key, float(v), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            else:
                # JAX scalar
                self.writer.add_scalar(key, float(value), step)

        self.writer.add_scalar("system/sps", self.algorithm.get_sps(), step)
        self.writer.add_scalar("system/episodes", self.algorithm.episode_count, step)

    # ── Evaluation ────────────────────────────────────────────

    def _evaluate_policy(self) -> dict[str, float]:
        eval_env = self.eval_env or self.algorithm.env
        print(f"\nEvaluating at step {self.algorithm.global_step}...")

        eval_metrics = self.algorithm.evaluate(
            eval_env=eval_env,
            num_episodes=self.eval_episodes,
            deterministic=self.eval_deterministic,
        )

        if self.writer is not None:
            step = self.algorithm.global_step
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, step)

        print(f"Eval mean return: {eval_metrics['mean_return']:.2f}")
        return eval_metrics

    # ── Checkpointing ─────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        step = self.algorithm.global_step
        path = f"{self.checkpoint_dir}/step_{step}"
        self.algorithm.save(path)
        self._checkpoint_paths.append(path)
        self._last_checkpoint_step = step
        if self.max_checkpoints and len(self._checkpoint_paths) > self.max_checkpoints:
            old = self._checkpoint_paths.popleft()
            shutil.rmtree(old, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────

    def _print_summary(self) -> None:
        total_time = time.time() - (self.start_time or time.time())

        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total timesteps: {self.algorithm.global_step}")
        print(f"Total episodes: {self.algorithm.episode_count}")
        print(f"Training time: {total_time:.2f}s")
        print(f"Average SPS: {self.algorithm.get_sps():.0f}")

        if self.algorithm.episode_returns:
            recent = self.algorithm.episode_returns[-100:]
            print(f"Final mean return (last {len(recent)} eps): {np.mean(recent):.2f}")

        print("=" * 50)
