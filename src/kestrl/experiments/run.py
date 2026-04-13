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
from importlib import import_module

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tensorboardX import SummaryWriter

logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

from kestrl.environments.registry import get_env_builder
from kestrl.environments.builders.brax_builder import BraxVectorEnv
from kestrl.trainers import Trainer, CompiledTrainer


def build_agent(env, cfg: DictConfig, seed: int):
    """Instantiate a class-based algorithm (SAC, PBSAC, ...) from Hydra config."""
    algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
    target = algo_cfg.pop('_target_')
    algo_cfg.pop('name', None)
    return instantiate({'_target_': target, '_recursive_': False}, env=env, algo_cfg=algo_cfg, seed=seed)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed      = cfg.experiment.seed
    env_id    = cfg.environment.env_id
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

    # ── Logging setup ─────────────────────────────────────────
    num_seeds   = cfg.experiment.get('num_seeds', 1)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_id_safe = re.sub(r'[/\\]', '-', env_id)
    seed_tag    = f"x{num_seeds}seeds" if num_seeds > 1 else f"seed{seed}"
    run_name    = f"{env_id_safe}_{algo_name}_{seed_tag}_{timestamp}"

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

    # ── Train — Fully compiled functional path ────────────────────────────────────
    if isinstance(env, BraxVectorEnv):
        algo_cfg = OmegaConf.to_container(cfg.algorithm, resolve=True)
        algo_cfg.pop('_target_', None)
        algo_cfg.pop('name', None)
        factory_path = algo_cfg.pop('compiled_factory')
        module_path, fn_name = factory_path.rsplit('.', 1)
        factory = getattr(import_module(module_path), fn_name)
        # Merge experiment-level logging/eval params into algo config
        algo_cfg['log_interval']  = cfg.experiment.get('log_interval', 50)
        algo_cfg['eval_interval'] = cfg.experiment.get('eval_interval', None)
        algo_cfg['eval_episodes'] = cfg.experiment.get('eval_episodes', 100)
        algo_cfg['env_id']        = cfg.environment.env_id

        trainer = CompiledTrainer(
            env,
            algo_cfg,
            fns_factory=factory,
            writer=writer,
            log_per_seed=num_seeds > 1,
        )
        if num_seeds > 1:
            trainer.train_seeds_live(num_seeds, seed=seed)
        else:
            trainer.train(seed=seed)

    # ── Train — class-based path (SAC, PBSAC, ...) ───────────
    else:
        agent = build_agent(env, cfg, seed=seed)
        agent.writer = writer

        ckpt_base = cfg.experiment.get('checkpoint_dir', None)
        trainer = Trainer(
            algorithm=agent,
            config={
                'total_timesteps':     cfg.algorithm.total_timesteps,
                'eval_interval':       cfg.experiment.get('eval_interval', None),
                'eval_episodes':       cfg.experiment.get('eval_episodes', 10),
                'checkpoint_dir':      f"{ckpt_base}/{run_name}" if ckpt_base else None,
                'checkpoint_interval': cfg.experiment.get('checkpoint_interval', None),
                'max_checkpoints':     cfg.experiment.get('max_checkpoints', None),
            },
            writer=writer,
        )
        trainer.train()

    env.close()
    writer.close()

    if cfg.experiment.get('track', False):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
