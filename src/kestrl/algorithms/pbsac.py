"""PAC-Bayes Soft Actor-Critic (PB-SAC).

Inherits SAC's infrastructure (networks, critics, replay buffer, optimisers,
checkpointing) and adds a BlockPosterior over actor parameters that is
optimised via a PAC-Bayes bound on policy generalisation.

Algorithm (Algorithm 1 of the paper):

  Each env step:
    1. Collect experience with posterior-guided UCB exploration.
    2. Standard SAC critic + actor update.
    3. Sync: posterior_mean ← current actor weights.
    4. Every pb_update_freq steps:
         a. Collect pb_rollout_trajectories fresh trajectories.
         b. Estimate τ_min from reward autocorrelation.
         c. Split 90/10 train/test.
         d. For pb_update_epochs epochs:
              - Sample M policies from the posterior.
              - Evaluate via importance sampling on training trajectories.
              - REINFORCE gradient on (posterior.mean, posterior.std, posterior.P).
              - Analytically update λ* = sqrt(C · KL + C').
         e. Compute certified bound on test trajectories and log.
         f. Inject: actor ← posterior_mean; freeze actor.
    5. Critic adaptation phase (actor frozen):
         - Average TD targets over adaptation_samples posterior samples.
         - Unfreeze after actor_freeze_steps.
    6. Every pb_reset_prior_freq steps:
         - EMA update of prior: μ₀ ← ι·υ + (1-ι)·μ₀.

Paper: https://arxiv.org/abs/2510.10544
Reference implementation: src/benchrl/algorithms/pbsac.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from kestrl.algorithms.sac import (
    SAC,
    soft_update,
    update_discrete,
    update_continuous_critics,
    update_continuous_actor_alpha
)
from kestrl.algorithms.functions.pbsac_updates import (
    update_pb_posterior,
    adaptive_update_discrete_critics,
    adaptive_update_continuous_critics
)
from kestrl.algorithms.functions.pbsac_losses import (
    compute_pac_bayes_bound,
    compute_posterior_guided_targets,
    get_continuous_actor_action_from_posterior,
    get_discrete_actor_action_from_posterior
)
from kestrl.distributions import (
    BlockPosterior,
    BlockPrior,
    kl_block,
    _construct_state_from_flat_state
)


@dataclass
class PBTrajectory:
    """One rollout used for PAC-Bayes evaluation."""
    states:      np.ndarray   # (H, obs_dim)
    actions:     np.ndarray   # (H,) discrete  or  (H, act_dim) continuous
    rewards:     np.ndarray   # (H,)
    log_probs_b: np.ndarray   # (H,)  log π_b(a_t | s_t)
    mask:        np.ndarray   # (H,)  True for active timesteps
    G:           float        # discounted return Σ γ^t r_t


class PBSAC(SAC):
    """PAC-Bayes Soft Actor-Critic."""

    def __init__(self, env, algo_cfg: dict[str, Any], *, seed: int = 0):
        super().__init__(env, algo_cfg, seed=seed)

        self.num_pb_envs             = self.config.get('pb_num_envs', 2)
        self.pb_rank                 = self.config.get('pb_rank', 10)
        self.pb_init_std             = self.config.get('pb_init_std', 0.01)
        self.delta                   = self.config.get('delta', 0.1)
        self.pb_update_freq          = self.config.get('pb_update_freq', 20_000)
        self.pb_update_epochs        = self.config.get('pb_update_epochs', 100)
        self.pb_rollout_trajectories = self.config.get('pb_rollout_trajectories', 100)
        self.pb_rollout_steps        = self.config.get('pb_rollout_steps', 500)
        self.pb_policy_samples       = int(self.config.get('pb_policy_samples', 16))
        self.pb_posterior_lr         = self.config.get('pb_posterior_lr', 3e-4)
        self.pb_reset_prior_freq     = self.config.get('pb_reset_prior_freq', 20_000)
        self.pb_prior_decay          = self.config.get('pb_prior_decay', 0.99)
        self.pac_bayes_active        = self.config.get('pac_bayes_active', True)
        self.adaptation_samples      = self.config.get('adaptation_samples', 256)
        self.actor_freeze_steps      = self.config.get('actor_freeze_steps', 20)
        self.explore_prob_init       = self.config.get('explore_prob_init', 0.5)
        self.explore_prob            = self.explore_prob_init
        self.explore_prob_final      = self.config.get('explore_prob_final', 0.1)
        self.explore_prob_decay_duration = self.config.get('explore_prob_decay_duration', 0.5)
        self.explore_n_samples       = self.config.get('explore_n_samples', 8)

        self.posterior = BlockPosterior.from_actor(
            self.actor, rank=self.pb_rank, init_std=self.pb_init_std
        )
        self.prior = BlockPrior.from_posterior(self.posterior)

        self.pb_lambda   = 1.0
        self.r_max       = self.config.get('r_max_estimate', 1.0)
        self.mixing_time = self.config.get('mixing_time', 1)

        self.actor_frozen       = False
        self.actor_freeze_until = 0

        self.pb_optimizer = nnx.Optimizer(
            self.posterior, optax.adam(self.pb_posterior_lr), wrt=nnx.Param
        )
        self.last_pb_metrics = {}

        from kestrl.environments.registry import get_env_builder
        env_builder = get_env_builder()

        env_id = None
        if hasattr(env, 'envs') and len(env.envs) > 0:
            single_env = env.envs[0]
            if hasattr(single_env, 'spec') and single_env.spec is not None:
                env_id = single_env.spec.id
            else:
                raise ValueError("Could not determine env_id from environment")

        try:
            self.pb_env = env_builder.build_env(
                env_id=env_id,
                num_envs=self.num_pb_envs,
                seed=42,
                capture_video=False,
                video_folder=None,
                wrappers=None,
            )
        except Exception as e:
            print(f"Warning: Could not create separate PB environment: {e}")
            self.pb_env = self.env

    def _sync_posterior_mean_from_actor(self) -> None:
        """Copy current actor weights into posterior.mean.

        Uses set_value rather than replacing the posterior object to preserve
        the optimizer's reference to the posterior's internal Variables.
        """
        params_tree = nnx.state(self.actor, nnx.Param)
        for path, param in nnx.to_flat_state(params_tree):
            self.posterior.layers[path].mean.set_value(param.get_value().flatten())

    def _inject_posterior_into_actor(self) -> None:
        """Set actor weights to the posterior mean (no noise added)."""
        flat_state = {name: lp.mean.get_value() for name, lp in self.posterior.layers.items()}
        actor_state = _construct_state_from_flat_state(self.actor, flat_state, self.posterior.shapes)
        graphdef, _ = nnx.split(self.actor)
        self.actor = nnx.merge(graphdef, actor_state)

    def _collect_pb_rollouts(self) -> tuple[list[PBTrajectory], list[PBTrajectory]]:
        """Collect trajectories with the current policy for PAC-Bayes evaluation.

        Returns a 90/10 train/test split. Each trajectory is padded to
        pb_rollout_steps with a boolean mask marking active timesteps.
        Updates self.r_max.
        """
        num_envs = self.pb_env.num_envs
        batches_needed = (self.pb_rollout_trajectories + num_envs - 1) // num_envs
        all_trajectories = []

        for _ in tqdm(range(batches_needed), desc="PB rollouts"):
            last_obs, _ = self.pb_env.reset()
            active = np.ones(num_envs, dtype=bool)
            env_lengths = np.zeros(num_envs, dtype=int)

            rewards_list, states_list, actions_list = [], [], []
            log_probs_list, active_mask_list = [], []

            for _ in range(self.pb_rollout_steps):
                obs_array = jnp.array(last_obs)
                actions, log_probs_b, _ = self.get_action(obs_array, deterministic=False)
                actions     = np.asarray(actions)
                log_probs_b = np.asarray(log_probs_b)

                next_obs, rewards, terminations, truncations, _ = self.pb_env.step(actions)
                self.r_max = max(self.r_max, float(np.max(np.abs(rewards))))

                states_list.append(np.asarray(obs_array))
                actions_list.append(actions)
                rewards_list.append(rewards)
                log_probs_list.append(log_probs_b)
                active_mask_list.append(active.copy())

                done = terminations | truncations
                for idx in range(num_envs):
                    if active[idx]:
                        env_lengths[idx] += 1
                        if done[idx]:
                            active[idx] = False

                last_obs = next_obs
                if not np.any(active):
                    break

            all_states    = np.array(states_list)
            all_actions   = np.array(actions_list)
            all_rewards   = np.array(rewards_list)
            all_log_probs = np.array(log_probs_list)

            for i in range(num_envs):
                H = env_lengths[i]
                if H == 0:
                    continue
                mask = np.zeros(self.pb_rollout_steps, dtype=bool)
                mask[:H] = True
                r = all_rewards[:, i][mask]
                G = float(sum(r[t] * (self.gamma ** t) for t in range(H)))
                all_trajectories.append(PBTrajectory(
                    states=all_states[:, i],
                    actions=all_actions[:, i],
                    rewards=all_rewards[:, i],
                    log_probs_b=all_log_probs[:, i],
                    mask=mask,
                    G=G,
                ))

        n_train = int(len(all_trajectories) * 0.9)
        return all_trajectories[:n_train], all_trajectories[n_train:]

    def _update_pac_bayes_components(
        self, train_trajs: list[PBTrajectory]
    ) -> dict[str, float]:
        """Optimise the posterior via the PAC-Bayes-λ surrogate.

        Runs pb_update_epochs epochs of alternating optimisation:
        gradient step on (μ, σ, P) for fixed λ, then analytical update
        λ* = sqrt(C · KL + C') for fixed posterior.

        Returns metrics from the final epoch.
        """
        if not self.pac_bayes_active:
            return {}

        key = self._next_key()
        original_params = nnx.state(self.actor, nnx.Param)

        H = self.max_ep_length
        self.episode_count += len(train_trajs)
        T = self.episode_count

        # ||c||² = R_max² · (1 - γ^{2H}) / (T · (1 - γ²))
        c_squared = max(
            1e-6,
            self.r_max ** 2 * (1 - self.gamma ** (2 * H))
            / (T * (1 - self.gamma ** 2))
        )
        C_const       = c_squared * self.mixing_time
        C_prime_const = C_const * math.log(math.sqrt(2) / self.delta)

        num_sampled = max(1, int(0.8 * len(train_trajs)))
        pb_metrics = {}
        for epoch in range(self.pb_update_epochs):
            indices = np.random.choice(len(train_trajs), num_sampled, replace=False)
            trajs   = [train_trajs[i] for i in indices]

            batch_data = {
                'states':      jnp.stack([t.states for t in trajs]),
                'actions':     jnp.stack([t.actions for t in trajs]),
                'log_probs_b': jnp.stack([t.log_probs_b for t in trajs]),
                'masks':       jnp.stack([t.mask for t in trajs]),
                'returns':     jnp.array([t.G for t in trajs]),
            }
            if not self.is_discrete:
                batch_data['action_scale'] = self.action_scale
                batch_data['action_bias']  = self.action_bias

            kl = kl_block(self.posterior, self.prior)
            self.pb_lambda = math.sqrt(C_const * kl + C_prime_const)

            pb_metrics = update_pb_posterior(
                self.posterior, self.prior, self.actor, self.pb_optimizer,
                self.is_discrete, key, self.pb_policy_samples,
                self.pb_lambda, C_const, C_prime_const, batch_data,
            )
            if epoch % max(self.pb_update_epochs // 5, 1) == 0:
                print(
                    f"  Epoch {epoch}/{self.pb_update_epochs}: "
                    f"loss={pb_metrics['loss']:.4f}  "
                    f"kl={pb_metrics['kl_div']:.4f}  "
                    f"λ={pb_metrics['lambda']:.4f}  "
                    f"return={pb_metrics['mean_empirical_return']:.4f}"
                )

        nnx.update(self.actor, original_params)
        return pb_metrics

    def _compute_pac_bayes_bound(
        self, test_trajs: list[PBTrajectory]
    ) -> dict[str, float]:
        """Evaluate the certified PAC-Bayes bound on held-out test trajectories.

        Bound (Theorem 1):
          E[V] ≥ Ê_ρ[R] − sqrt( ||c||² · τ · (KL(ρ‖μ) + ln(√2/δ)) )

        Returns a dict with keys: certified_return, empirical_return,
        uncertainty_term, kl_div, c_squared, mixing_time, r_max.
        """
        batch_data = {
            'states':      jnp.stack([t.states for t in test_trajs]),
            'actions':     jnp.stack([t.actions for t in test_trajs]),
            'log_probs_b': jnp.stack([t.log_probs_b for t in test_trajs]),
            'masks':       jnp.stack([t.mask for t in test_trajs]),
            'returns':     jnp.array([t.G for t in test_trajs]),
        }
        if not self.is_discrete:
            batch_data['action_scale'] = self.action_scale
            batch_data['action_bias']  = self.action_bias

        pb_metrics = compute_pac_bayes_bound(
            self.posterior, self.prior, self.actor, self.is_discrete,
            self._next_key(), self.pb_policy_samples,
            max(1, self.episode_count), self.max_ep_length,
            self.r_max, self.mixing_time, self.gamma, self.delta,
            batch_data,
        )
        result = {k: float(v) for k, v in pb_metrics.items()}
        print(
            f"PAC-Bayes bound: certified={result['certified_return']:.4f}  "
            f"empirical={result['empirical_return']:.4f}  "
            f"uncertainty={result['uncertainty_term']:.4f}"
        )
        return result

    def _estimate_mixing_time(self, trajectories: list[PBTrajectory]) -> int:
        """Estimate the Markov chain mixing time from trajectory autocorrelations.

        For each trajectory, finds the first lag where the reward autocorrelation
        drops below 0.2, applies a 1.5× safety margin, and takes the max over
        all trajectories. The estimate is monotonically non-decreasing (conservative).
        """
        mt = 1
        candidates = []
        cross_val = getattr(self, 'cross_validation', False)

        for traj in trajectories:
            valid_len = int(np.sum(traj.mask))
            if valid_len < 10:
                continue

            signals = [traj.rewards[:valid_len]]
            if cross_val and traj.states is not None:
                for feat in traj.states[:valid_len].T:
                    signals.append(feat)

            for signal in signals:
                signal = signal - signal.mean()
                if np.var(signal) < 1e-8:
                    continue

                n   = len(signal)
                acf = np.correlate(signal, signal, mode='full')
                if abs(acf[n - 1]) < 1e-8:
                    continue

                acf   = acf[n - 1:] / acf[n - 1]
                below = np.where(acf < 0.2)[0]
                tmp   = below[0] if len(below) > 0 else (n - 1)
                candidates.append(int(tmp * 1.5) if tmp > 1 else 1)

        if candidates:
            mt = max(mt, max(candidates))

        self.mixing_time = max(getattr(self, 'mixing_time', 1), mt)
        return self.mixing_time

    def get_action(self, obs: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Select an action, optionally using posterior-guided UCB exploration."""
        key = self._next_key()

        if self.is_discrete:
            if deterministic:
                return jnp.argmax(self.actor(obs), axis=-1), None, None
            return get_discrete_actor_action_from_posterior(
                self.posterior, self.actor, self.critic1, self.critic2,
                obs, self.explore_prob, self.explore_n_samples, key,
            )

        if deterministic:
            logits = self.actor(obs)
            mean = jnp.tanh(logits['mean']) * self.action_scale + self.action_bias
            return mean, None, None

        return get_continuous_actor_action_from_posterior(
            self.posterior, self.actor, self.critic1, self.critic2,
            obs, self.action_scale, self.action_bias,
            self.explore_prob, self.explore_n_samples, key,
        )

    def train_step(self) -> dict[str, float]:
        """One PB-SAC training step."""
        metrics = {}

        if self.actor_frozen and self.global_step >= self.actor_freeze_until:
            self.actor_frozen = False

        metrics.update(self.collect_rollouts(self.train_freq))

        if self.global_step < self.learning_starts:
            return metrics

        self.explore_prob = max(
            self.explore_prob_final,
            self.explore_prob_init - (self.global_step - self.learning_starts)
                  / max(1, self.total_timesteps - self.learning_starts) * self.explore_prob_decay_duration
        )

        data = self.replay_buffer.sample(self.batch_size)
        key  = self._next_key()

        if self.actor_frozen:
            target_q = compute_posterior_guided_targets(
                self.posterior, self.actor,
                self.target_critic1, self.target_critic2, self.log_alpha,
                data.next_observations, data.rewards, data.dones,
                self.gamma, self.adaptation_samples, key, self.is_discrete,
                self.action_scale, self.action_bias,
            )
            if self.is_discrete:
                sac_metrics = adaptive_update_discrete_critics(
                    self.critic1, self.critic2,
                    self.critic1_optimizer, self.critic2_optimizer,
                    data.observations, data.actions, target_q,
                )
            else:
                sac_metrics = adaptive_update_continuous_critics(
                    self.critic1, self.critic2,
                    self.critic1_optimizer, self.critic2_optimizer,
                    data.observations, data.actions, target_q,
                )
        else:
            if self.is_discrete:
                sac_metrics = update_discrete(
                    self.actor, self.critic1, self.critic2,
                    self.target_critic1, self.target_critic2,
                    self.actor_optimizer, self.critic1_optimizer, self.critic2_optimizer,
                    self.log_alpha, self.alpha_optimizer,
                    data.observations, data.actions, data.rewards,
                    data.next_observations, data.dones,
                    self.gamma, self.target_entropy, key,
                )
            else:
                sac_metrics = update_continuous_critics(
                    self.actor, self.critic1, self.critic2,
                    self.target_critic1, self.target_critic2,
                    self.critic1_optimizer, self.critic2_optimizer,
                    data.observations, data.actions, data.rewards,
                    data.next_observations, data.dones,
                    self.log_alpha, self.gamma,
                    self.action_scale, self.action_bias, key,
                )
                if self.global_step % self.policy_frequency == 0:
                    for _ in range(self.policy_frequency):
                        key = self._next_key()
                        actor_alpha_metrics = update_continuous_actor_alpha(
                            self.actor, self.critic1, self.critic2,
                            self.actor_optimizer, self.log_alpha, self.alpha_optimizer,
                            data.observations, self.action_scale, self.action_bias,
                            self.autotune_alpha, self.target_entropy, key,
                        )
                        sac_metrics.update(actor_alpha_metrics)

        metrics.update(sac_metrics)

        if self.global_step % self.target_update_interval == 0:
            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

        self._sync_posterior_mean_from_actor()

        pb_active = (
            self.global_step > 0
            and self.global_step % self.pb_update_freq == 0
            and self.global_step != self.learning_starts
        )
        if pb_active:
            t0 = time.time()
            train_trajs, test_trajs = self._collect_pb_rollouts()
            self.mixing_time = self._estimate_mixing_time(train_trajs + test_trajs)

            if train_trajs and test_trajs:
                pb_info = self._update_pac_bayes_components(train_trajs)
                self.last_pb_metrics.update(
                    {f'pac_bayes/update/{k}': v for k, v in pb_info.items()}
                )
                bound_info = self._compute_pac_bayes_bound(test_trajs)
                self.last_pb_metrics.update(
                    {f'pac_bayes/{k}': v for k, v in bound_info.items()}
                )
                self._inject_posterior_into_actor()
                self.actor_frozen       = True
                self.actor_freeze_until = self.global_step + self.actor_freeze_steps

            self.last_pb_metrics['pac_bayes/time_s'] = time.time() - t0

        if self.global_step % self.pb_reset_prior_freq == 0:
            self.prior = self.prior.ema_update(self.posterior, self.pb_prior_decay)

        metrics['pac_bayes/r_max']        = self.r_max
        metrics['pac_bayes/mixing_time']  = self.mixing_time
        metrics['pac_bayes/actor_frozen'] = int(self.actor_frozen)
        metrics['explore_prob']           = self.explore_prob
        metrics.update(self.last_pb_metrics)

        return metrics

    def save(self, path: str) -> None:
        # TODO: persist posterior, prior, r_max, mixing_time
        super().save(path)

    def load(self, path: str) -> None:
        # TODO: restore posterior, prior, r_max, mixing_time
        super().load(path)
