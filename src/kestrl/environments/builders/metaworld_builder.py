from typing import Any, Dict, List, Optional, Callable
import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, AutoresetMode

from ..base import EnvironmentFactory


class MetaWorldEnvironmentFactory(EnvironmentFactory):
    """Factory for MetaWorld environments."""
    
    def __init__(self):
        try:
            import metaworld
            self.metaworld = metaworld
            self._ml1 = metaworld.ML1
            self._mt1 = metaworld.MT1
            self._ml10 = metaworld.ML10
            self._mt10 = metaworld.MT10
            self._ml45 = metaworld.ML45
            self._mt50 = metaworld.MT50
        except ImportError:
            raise ImportError("MetaWorld not installed. Install with: pip install metaworld")
    
    def make_single_env(
        self,
        env_id: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ) -> gym.Env:
        """Create a single MetaWorld environment."""
        
        # Parse environment type and task
        env_type, task_name = self._parse_env_id(env_id)
        
        if env_type == "ml1":
            benchmark = self._ml1(task_name, seed=seed)
            env = benchmark.train_classes[task_name]()
            env.set_task(benchmark.train_tasks[0])
        elif env_type == "mt1":
            env = self.metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name + "-goal-observable"]()
            if seed:
                env.seed(seed)
        else:
            raise ValueError(f"Unsupported MetaWorld environment type: {env_type}")
        
        # Wrap to make it compatible with Gymnasium API
        from ..wrappers.api_wrappers import MetaWorldToGymWrapper
        env = MetaWorldToGymWrapper(env, render_mode=render_mode)
        
        return env
    
    def make_vector_env(
        self,
        env_id: str,
        num_envs: int,
        seed: Optional[int] = None,
        capture_video: bool = False,
        video_folder: Optional[str] = None,
        wrappers: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> gym.vector.VectorEnv:
        """Create vectorized MetaWorld environments."""
        
        def make_env(idx: int) -> Callable[[], gym.Env]:
            def _thunk() -> gym.Env:
                render_mode = 'rgb_array' if capture_video and idx == 0 else None
                env = self.make_single_env(
                    env_id,
                    seed=seed + idx if seed else None,
                    render_mode=render_mode,
                    **kwargs
                )
                
                # Add video recording for first environment
                if capture_video and idx == 0 and video_folder:
                    env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
                
                # Apply standard wrappers
                env = gym.wrappers.RecordEpisodeStatistics(env)
                
                # Apply custom wrappers
                if wrappers:
                    from hydra.utils import instantiate
                    for wrapper_cfg in wrappers:
                        env = instantiate(wrapper_cfg, env=env)
                
                return env
            
            return _thunk
        
        env_fns = [make_env(i) for i in range(num_envs)]
        return SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    
    def get_available_envs(self) -> List[str]:
        """Get list of available MetaWorld environments."""
        ml1_envs = [f"ml1-{name}" for name in self.metaworld.ML1.ENV_NAMES]
        mt1_envs = [f"mt1-{name}" for name in self.metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()]
        return ml1_envs + mt1_envs
    
    def validate_env_id(self, env_id: str) -> bool:
        """Check if environment ID is valid for MetaWorld."""
        return env_id in self.get_available_envs()
    
    def _parse_env_id(self, env_id: str) -> tuple:
        """Parse MetaWorld environment ID."""
        parts = env_id.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MetaWorld environment ID: {env_id}")
        return parts[0], parts[1]