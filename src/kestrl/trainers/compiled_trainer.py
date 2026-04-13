"""Training controller for compiled (lax.scan) algorithms.

    trainer = CompiledTrainer(env, config, fns_factory=make_compiled_sac, writer=writer)
    carry = trainer.train(seed=0)                              # single seed, live progress
    all_carries, _ = trainer.train_seeds_live(num_seeds=8)    # multi-seed, vmapped epochs

The algorithm is injected via fns_factory — any callable (env, config) -> NamedTuple
with (train, init, step_epoch, evaluate) fields works. The factory is resolved from
the compiled_factory key in the algorithm YAML by run.py.
"""

import time
from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from kestrl.environments.builders.brax_builder import BraxVectorEnv


class CompiledTrainer:
    """Algorithm-agnostic training controller for compiled (lax.scan) routes.

    writer accepts any object with add_scalar() (TensorboardX, wandb.run, etc.).
    log_per_seed writes per-seed curves under "seedN/key" alongside the cross-seed mean.
    """

    def __init__(
        self,
        env: BraxVectorEnv,
        config: dict[str, Any],
        fns_factory: Callable,
        *,
        writer=None,
        eval_env: BraxVectorEnv | None = None,
        log_per_seed: bool = False,
    ) -> None:
        self.env          = env
        self.config       = config
        self.writer       = writer
        self.eval_env     = eval_env or env
        self.log_per_seed = log_per_seed

        self._fns = fns_factory(env, config)

        self._train_freq      = config.get('train_freq', 1)
        self._num_envs        = env.num_envs
        self._total_timesteps = config.get('total_timesteps', 1_000_000)
        self._log_interval    = config.get('log_interval', 50)
        self._num_train_steps = self._total_timesteps // (self._train_freq * self._num_envs)
        self._num_epochs      = self._num_train_steps // self._log_interval
        self._eval_interval   = config.get('eval_interval', None)  # env steps
        self._eval_episodes   = config.get('eval_episodes', 100)

        self.start_time: float | None = None

    # ── Single-seed training ──────────────────────────────────────────────────

    def train(self, seed: int = 0):
        """Single-seed training. Python epoch loop over step_epoch with live tqdm."""
        self.start_time = time.time()
        key   = jax.random.PRNGKey(seed)
        carry = jax.jit(self._fns.init)(key)

        steps_per_epoch    = self._log_interval * self._train_freq * self._num_envs
        total_env_steps    = self._num_epochs * steps_per_epoch
        last_eval_step     = 0
        last_return        = 0.0
        last_critic_loss   = 0.0

        print(f"Compiled — single seed {seed}")
        print(f"  {self._num_epochs} epochs × {self._log_interval} steps"
              f" × {self._train_freq * self._num_envs} env-steps/step"
              f" = {total_env_steps:,} env steps")
        print(f"  log_interval: {self._log_interval} train steps = {steps_per_epoch:,} env steps")
        print("-" * 50)

        with tqdm.tqdm(total=total_env_steps, desc="Training", unit="env-step") as pbar:
            for _ in range(self._num_epochs):
                carry, epoch_metrics = self._fns.step_epoch(carry)
                global_step = int(carry.global_step)

                # epoch_metrics[k] has shape (log_interval,) — average over window
                mean_metrics = {k: float(v.mean()) for k, v in epoch_metrics.items()}
                last_critic_loss = mean_metrics.get('critic/loss', last_critic_loss)
                last_return      = mean_metrics.get('episode/return', last_return)

                self._log_metrics(mean_metrics, global_step)

                pbar.update(steps_per_epoch)
                pbar.set_postfix({
                    'return':      f'{last_return:.2f}',
                    'critic_loss': f'{last_critic_loss:.4f}',
                    'SPS':         f'{self._sps(global_step):.0f}',
                })

                # Periodic eval
                if (self._eval_interval is not None
                        and global_step >= last_eval_step + self._eval_interval):
                    result = self._fns.evaluate(
                        carry.bundle_state, self.eval_env,
                        num_episodes=self._eval_episodes,
                    )
                    last_return    = result['mean_return']
                    last_eval_step = global_step
                    self._log_eval(result, global_step)
                    print(f"\n  eval @ {global_step:,}: return={last_return:.2f}"
                          f"  ±{result['std_return']:.2f}")

        self._print_summary(carry, seed=seed)
        return carry

    # ── Multi-seed vmap training ──────────────────────────────────────────────

    def train_seeds_live(
        self,
        num_seeds: int,
        seed: int = 0,
    ) -> tuple[Any, dict]:
        """Multi-seed training via vmapped epoch loop. One D2H sync per epoch.

        all_carries has a leading num_seeds axis.
        all_metrics[k] is numpy of shape (num_seeds, num_train_steps).
        """
        self.start_time = time.time()
        keys = jax.random.split(jax.random.PRNGKey(seed), num_seeds)

        steps_per_epoch = self._log_interval * self._train_freq * self._num_envs
        total_env_steps = self._num_epochs * steps_per_epoch

        print(f"Compiled — {num_seeds} seeds (vmapped epochs)")
        print(f"  {self._num_epochs} epochs × {steps_per_epoch:,} env-steps/epoch per seed")
        print(f"  Compiling first epoch…")

        all_carries  = jax.jit(jax.vmap(self._fns.init))(keys)
        vmapped_step = self._fns.vmap_step_epoch

        collected: list[dict] = []   # per epoch: {k: ndarray (num_seeds, log_interval)}
        last_critic_loss = 0.0
        last_return       = 0.0
        last_eval_step    = 0

        with tqdm.tqdm(total=total_env_steps, unit='env-step',
                       desc=f'{num_seeds} seeds') as pbar:
            for _ in range(self._num_epochs):
                all_carries, epoch_metrics = vmapped_step(all_carries)
                # epoch_metrics[k]: (num_seeds, log_interval) JAX arrays

                step = int(all_carries.global_step[0])

                # One D2H per metric key — averaged over seeds × log_interval
                mean_m = {k: float(v.mean()) for k, v in epoch_metrics.items()}
                last_critic_loss = mean_m.get('critic/loss', last_critic_loss)
                last_return      = mean_m.get('episode/return', last_return)

                self._log_metrics(mean_m, step)

                if self.log_per_seed and self.writer is not None:
                    for si in range(num_seeds):
                        for k, v in epoch_metrics.items():
                            self.writer.add_scalar(
                                f'seed{si}/{k}', float(v[si].mean()), step,
                            )

                collected.append({k: np.asarray(v) for k, v in epoch_metrics.items()})

                # Periodic eval — runs for each seed, logs cross-seed mean return
                if (self._eval_interval is not None
                        and step >= last_eval_step + self._eval_interval):
                    seed_returns = []
                    for si in range(num_seeds):
                        state_i = jax.tree.map(lambda x: x[si], all_carries.bundle_state)
                        result  = self._fns.evaluate(
                            state_i, self.eval_env, num_episodes=self._eval_episodes,
                        )
                        prefix = f'seed{si}/eval' if num_seeds > 1 else 'eval'
                        self._log_eval(result, step, prefix=prefix)
                        seed_returns.append(result['mean_return'])
                    last_return    = float(np.mean(seed_returns))
                    last_eval_step = step

                pbar.update(steps_per_epoch)
                pbar.set_postfix({
                    'return':      f'{last_return:.2f}',
                    'critic/loss': f'{last_critic_loss:.4f}',
                    'SPS':         f'{self._sps(step):.0f}',
                })

        # Rebuild (num_seeds, num_train_steps) numpy arrays for return value
        all_metrics: dict = {
            k: np.concatenate([e[k] for e in collected], axis=1)
            for k in collected[0]
        }

        # Post-training evaluation
        print(f"  Evaluating {num_seeds} seeds…")
        for seed_i in range(num_seeds):
            state_i = jax.tree.map(lambda x: x[seed_i], all_carries.bundle_state)
            result  = self._fns.evaluate(
                state_i, self.eval_env, num_episodes=self._eval_episodes,
            )
            prefix = f'seed{seed_i}/eval' if num_seeds > 1 else 'eval'
            self._log_eval(result, total_env_steps, prefix=prefix)
            print(f"    seed {seed_i}: mean_return={result['mean_return']:.2f}"
                  f"  ±{result['std_return']:.2f}")

        if self.writer is not None:
            self.writer.flush()

        self._print_seeds_summary(all_carries, all_metrics, num_seeds)
        return all_carries, all_metrics

    # ── [Reference] Single-dispatch multi-seed ────────────────────────────────
    # Kept as dead code. Fast for small num_train_steps (e.g. num_envs=2048) but
    # has no progress output during the run — the full training is one XLA kernel.
    # Use train_seeds_live for num_envs=1 or any long-running configuration.

    def train_seeds(
        self,
        num_seeds: int,
        seed: int = 0,
    ) -> tuple[Any, dict]:
        """Single-dispatch vmap: entire training in one jit(vmap(train)) call.

        No progress during the run. Use train_seeds_live for long runs.
        """
        self.start_time = time.time()
        keys = jax.random.split(jax.random.PRNGKey(seed), num_seeds)

        print(f"Compiled — {num_seeds} seeds (vmap), compiling…")
        t0 = time.time()

        all_carries, all_metrics = jax.jit(jax.vmap(self._fns.train))(keys)
        # block_until_ready forces the computation to complete before timing
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), all_metrics)

        elapsed = time.time() - t0
        total_env_steps = self._num_train_steps * self._train_freq * self._num_envs
        sps_total = (num_seeds * total_env_steps) / elapsed
        print(f"  Done in {elapsed:.1f}s — {sps_total:,.0f} env-steps/s across all seeds")

        self._log_vmap_metrics(all_metrics, num_seeds)

        # Post-training evaluation — one eval per seed (eval during vmap is not feasible)
        total_env_steps_all = self._num_train_steps * self._train_freq * self._num_envs
        print(f"  Evaluating {num_seeds} seeds…")
        for seed_i in range(num_seeds):
            state_i = jax.tree.map(lambda x: x[seed_i], all_carries.bundle_state)
            result  = self._fns.evaluate(
                state_i, self.eval_env,
                num_episodes=self._eval_episodes,
            )
            prefix = f'seed{seed_i}/eval' if num_seeds > 1 else 'eval'
            self._log_eval(result, total_env_steps_all, prefix=prefix)
            print(f"    seed {seed_i}: mean_return={result['mean_return']:.2f}"
                  f"  ±{result['std_return']:.2f}")

        if self.writer is not None:
            self.writer.flush()

        self._print_seeds_summary(all_carries, all_metrics, num_seeds)
        return all_carries, all_metrics

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        bundle_state,
        num_episodes: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """Evaluate a trained bundle_state. Wraps fns.evaluate."""
        result = self._fns.evaluate(
            bundle_state,
            self.eval_env,
            num_episodes=num_episodes or self._eval_episodes,
            seed=seed,
        )
        print(f"Eval: mean={result['mean_return']:.2f} ±{result['std_return']:.2f}"
              f"  median={result['median_return']:.2f}  ({result['num_episodes']} eps)")
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sps(self, global_step: int) -> float:
        if self.start_time is None or global_step == 0:
            return 0.0
        return global_step / (time.time() - self.start_time)

    def _log_metrics(self, metrics: dict, step: int) -> None:
        if self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        self.writer.add_scalar('system/sps', self._sps(step), step)

    def _log_eval(self, result: dict, step: int, prefix: str = 'eval') -> None:
        if self.writer is None:
            return
        for k, v in result.items():
            self.writer.add_scalar(f'{prefix}/{k}', v, step)

    def _log_vmap_metrics(self, all_metrics: dict, num_seeds: int) -> None:
        """Log post-hoc metrics from train_seeds at epoch resolution.

        One D2H per key. Subsampled to at most 1000 log points.
        """
        if self.writer is None:
            return

        steps_per_epoch = self._log_interval * self._train_freq * self._num_envs
        usable_steps    = self._num_epochs * self._log_interval
        # Cap log points at 1000 regardless of num_epochs
        stride = max(1, self._num_epochs // 1000)

        for k, v in all_metrics.items():
            # One D2H transfer per metric key
            v_np    = np.asarray(v[:, :usable_steps])                     # (S, E*L)
            v_epoch = v_np.reshape(num_seeds, self._num_epochs, self._log_interval)
            v_mean  = v_epoch.mean(axis=-1)                               # (S, E) numpy

            for epoch_i in range(stride - 1, self._num_epochs, stride):
                step = (epoch_i + 1) * steps_per_epoch
                self.writer.add_scalar(k, float(v_mean[:, epoch_i].mean()), step)

                if self.log_per_seed:
                    for seed_i in range(num_seeds):
                        self.writer.add_scalar(
                            f'seed{seed_i}/{k}', float(v_mean[seed_i, epoch_i]), step
                        )

        self.writer.flush()

    def _print_summary(self, carry: Any, seed: int) -> None:
        total_time = time.time() - (self.start_time or time.time())
        global_step = int(carry.global_step)
        print(f"\n{'='*50}")
        print(f"Training complete — seed {seed}")
        print(f"  env steps : {global_step:,}")
        print(f"  wall time : {total_time:.1f}s")
        print(f"  avg SPS   : {self._sps(global_step):,.0f}")
        print(f"{'='*50}")

    def _print_seeds_summary(
        self,
        all_carries: Any,
        all_metrics: dict,
        num_seeds: int,
    ) -> None:
        total_time = time.time() - (self.start_time or time.time())
        # Cross-seed mean of last epoch's critic loss
        last_critic = float(all_metrics.get('critic/loss', jnp.zeros((num_seeds, 1)))[:, -1].mean())
        print(f"\n{'='*50}")
        print(f"Multi-seed training complete — {num_seeds} seeds")
        print(f"  wall time       : {total_time:.1f}s")
        print(f"  final critic/loss (mean across seeds): {last_critic:.4f}")
        print(f"  Call evaluate(all_carries.bundle_state[i]) for per-seed eval")
        print(f"{'='*50}")
