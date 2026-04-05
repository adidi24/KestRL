# KestRL (Kestrel)

KestRL is a JAX/Flax reinforcement learning library built around one idea: gradient updates should never wait on Python. It's designed for clean algorithm implementations, reproducible research, and easy extension.

Networks are defined with Flax NNX; full OOP, readable, composable. Before training starts, the entire model bundle is split into a static graph definition and a mutable state pytree. The graph definition is captured in a closure; only the state flows in and out of JIT. The result is zero retracing, zero Python overhead per gradient step.

For Brax and MJX environments, this extends further: the rollout, buffer write, gradient update, and soft target update all fold into a single `lax.scan` kernel. Multi-seed training becomes `jax.vmap` over independent carries, multiple seeds run in one dispatch, with one host sync per epoch for logging.

---

## Two routes

### Classic (Gymnasium)

Standard Gymnasium environments, numpy replay buffer, JIT-compiled gradient updates. The env.step crosses Python; everything else doesn't.

```python
agent = SAC(env, algo_cfg, seed=0)
trainer = Trainer(agent, config, writer=writer)
trainer.train()
```

### Compiled (Brax, MJX)

Entire training loop on device — rollout, buffer write, gradient update, soft target update, all in one `lax.scan` kernel.

```python
trainer = CompiledTrainer(env, config, writer=writer)
trainer.train(seed=0)                           # single seed, live progress
trainer.train_seeds_live(num_seeds=10, seed=0)  # 10 seeds, vmapped epochs
```

Under the hood, both routes share the same frozen-bundle pattern:

```python
# define with NNX  readable, standard OOP
self.actor  = MultiHeadMLP(obs_dim, head_configs, hidden_dims, rngs=rngs)
self.critic = MLP(critic_in, 1, hidden_dims, rngs=rngs)

# split once before training
bundle_gd, bundle_state = nnx.split(bundle)

# compile a pure update — bundle_gd is captured, never re-traced
self._jit_update = _make_frozen_update(bundle_gd, ...)

# each step: only state crosses the boundary
bundle_state, metrics = self._jit_update(bundle_state, batch, key)
```

In the compiled route, `bundle_state` becomes a field in the `lax.scan` carry. Same pattern, different scope.

---

## Installation

```bash
git clone https://github.com/adidi24/KestRL.git
cd KestRL
uv sync
```

Python 3.10–3.12. Ships with the JAX CPU backend; swap in `jax[cuda12]` or `jax[tpu]` for accelerators.

## Quick start

```bash
# SAC on HalfCheetah-v5
uv run python -m kestrl.experiments.run

# SAC on CartPole-v1 (discrete)
uv run python -m kestrl.experiments.run algorithm=sac_cartpole environment=cartpole

# Compiled SAC on Brax Ant, 4 seeds
uv run python -m kestrl.experiments.run environment=brax algorithm=sac_mujoco experiment.num_seeds=4

# W&B tracking
uv run python -m kestrl.experiments.run experiment.track=true
```

Configuration via [Hydra](https://hydra.cc), config groups under `src/kestrl/configs/`.

## Algorithms

| Algorithm | Discrete | Continuous | Route |
| --------- | -------- | ---------- | ----- |
| SAC | ✔ | ✔ | Classic + Compiled |
| PB-SAC | ✔ | ✔ | Classic |
| PPO | — | — | Planned |

## Alternatives

**JAX**
- [rejax](https://github.com/keraJLi/rejax) — same compiled-training idea, broader algorithm coverage (PPO, DQN, TD3, IQN, PQN), requires JAX-native environments
- [PureJaxRL](https://github.com/luchris429/purejaxrl) — where the end-to-end JAX training idea started, single-file and easy to read
- [Stoix](https://github.com/EdanToledo/Stoix) — distributed actor-learner on TPU/GPU clusters, serious scale

**PyTorch**
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file implementations, the best place to read how an algorithm actually works
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — the production standard

## License

MIT
