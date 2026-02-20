"""KestRL — Hydra training entry point.

Usage:
    # Defaults (SAC on Pendulum-v1)
    python -m kestrl.experiments.run

    # Override algorithm and environment
    python -m kestrl.experiments.run algorithm=sac_cartpole environment=cartpole

    # Override individual params
    python -m kestrl.experiments.run algorithm=sac_mujoco experiment.seed=42 experiment.track=true

    # Multi-run (seed sweep)
    python -m kestrl.experiments.run -m experiment.seed=1,2,3 algorithm=sac_mujoco
"""

import os
import re
import logging
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tensorboardX import SummaryWriter

logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

from kestrl.environments.registry import get_env_builder
from kestrl.trainers import Trainer


def build_agent(env, cfg: DictConfig, seed: int):
    """Instantiate algorithm from Hydra config.

    Pops _target_ and name from the algorithm config, passes the rest
    as a plain dict to the algorithm constructor.
    """
    algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    target = algo_cfg.pop('_target_')
    algo_cfg.pop('name', None)
    return instantiate({'_target_': target, '_recursive_': False}, env=env, algo_cfg=algo_cfg, seed=seed)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.experiment.seed
    env_id = cfg.environment.env_id
    algo_name = cfg.algorithm.get('name', 'algorithm')

    # ── Environment ───────────────────────────────────────────
    env_builder = get_env_builder()
    env = env_builder.build_env(
        env_id=env_id,
        num_envs=cfg.environment.get('num_envs', 1),
        seed=seed,
        capture_video=cfg.experiment.get('capture_video', False),
        video_folder=cfg.experiment.get('video_folder', None),
        wrappers=list(cfg.environment.get('wrappers', [])) or None,
    )

    # ── Agent ─────────────────────────────────────────────────
    agent = build_agent(env, cfg, seed=seed)

    # ── Logging ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_id_safe = re.sub(r'[/\\]', '-', env_id)
    run_name = f"{env_id_safe}_{algo_name}_seed{seed}_{timestamp}"

    if cfg.experiment.get('track', False):
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'kestrl'),
            entity=os.getenv('WANDB_ENTITY', None),
            group=f"{algo_name}_{env_id_safe}",
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(
            f"|{k}|{v}|"
            for k, v in OmegaConf.to_container(cfg.algorithm).items()
            if not isinstance(v, dict)
        ),
    )
    agent.writer = writer

    # ── Train ─────────────────────────────────────────────────
    ckpt_base = cfg.experiment.get('checkpoint_dir', None)
    trainer = Trainer(
        algorithm=agent,
        config={
            'total_timesteps': cfg.algorithm.total_timesteps,
            'eval_interval': cfg.experiment.get('eval_interval', None),
            'eval_episodes': cfg.experiment.get('eval_episodes', 10),
            'checkpoint_dir': f"{ckpt_base}/{run_name}" if ckpt_base else None,
            'checkpoint_interval': cfg.experiment.get('checkpoint_interval', None),
            'max_checkpoints': cfg.experiment.get('max_checkpoints', None),
        },
        writer=writer,
    )
    trainer.train()

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
