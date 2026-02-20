# KestRL (Kestrel)

A modular reinforcement learning library built on [JAX](https://github.com/google/jax) and [Flax NNX](https://flax.readthedocs.io/en/latest/). Designed for clean algorithm implementations, reproducible research, and easy extension.

## Architecture

KestRL is organized around five decoupled layers:

```
Environments   →   clean factory/wrapper system, Hydra-composable
Buffers        →   ReplayBuffer (off-policy), RolloutBuffer (on-policy, planned)
Networks       →   MLP, MultiHeadMLP, ...
Algorithms     →   BaseAlgorithm interface: train_step / get_action / evaluate / save / load
Trainer        →   training loop, logging, evaluation — independent of algorithm
```

Algorithms only implement `train_step()`. The Trainer handles the loop, progress tracking, TensorBoard/W&B logging, and evaluation. Networks are configured via Hydra and instantiated with `_partial_=True`, so switching architectures (MLP → CNN) requires only a YAML change.

## Installation

```bash
git clone https://github.com/abdelkrimzitouni/KestRL.git
cd KestRL
uv sync
```

Requires Python 3.10–3.12. JAX CPU backend is installed by default.

## Quick Start

```bash
# SAC on Pendulum-v1 (continuous)
python -m kestrl.experiments.run

# SAC on CartPole-v1 (discrete)
python -m kestrl.experiments.run algorithm=sac_cartpole environment=cartpole

# Custom overrides
python -m kestrl.experiments.run algorithm=sac_mujoco experiment.seed=42 experiment.track=true

# Seed sweep (Hydra multirun)
python -m kestrl.experiments.run -m experiment.seed=1,2,3 algorithm=sac_mujoco
```

## Configuration

Experiments are configured via [Hydra](https://hydra.cc). Config groups:

```
src/kestrl/configs/
├── config.yaml                  # root: composes algorithm + environment + experiment
├── algorithm/
│   ├── sac_mujoco.yaml          # SAC for continuous control
│   └── sac_cartpole.yaml        # SAC for discrete control
├── environment/
│   ├── mujoco.yaml
│   └── cartpole.yaml
└── experiment/
    └── base.yaml                # seed, tracking, eval, checkpoint settings
```

Network architecture is specified inside the algorithm config and instantiated at runtime:

```yaml
# algorithm/sac_mujoco.yaml
actor_network:
  _target_: kestrl.networks.MultiHeadMLP
  hidden_dims: [256, 256]
  activation: relu
```

Switching to a CNN actor later requires only changing `_target_` — no algorithm code changes.

## Implemented Algorithms

| Algorithm | Action space | Status |
|-----------|-------------|--------|
| SAC | Discrete + Continuous | Done |
| PPO | Discrete + Continuous | Planned |
| DQN | Discrete | Planned |
| PBSAC (PAC-Bayes SAC) | Discrete + Continuous | In progress |

## Logging

TensorBoard is enabled by default. W&B can be enabled via `experiment.track=true` (requires `WANDB_PROJECT` and `WANDB_ENTITY` in your `.env`).

```bash
tensorboard --logdir runs/
```

## License

MIT
