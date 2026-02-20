import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode
from typing import Any, Dict, List, Optional, Callable
from hydra.utils import instantiate

from ..base import EnvironmentFactory


class GymEnvironmentFactory(EnvironmentFactory):
    """Factory for Gymnasium environments."""

    def __init__(self, async_envs: bool = False):
        self.async_envs = async_envs
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pass
    
    def make_single_env(
        self,
        env_id: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ) -> gym.Env:
        """Create a single Gym environment."""
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        
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
        """Create vectorized Gym environments."""
        
        def make_env(idx: int) -> Callable[[], gym.Env]:
            def _thunk() -> gym.Env:
                # Create environment
                if capture_video and idx == 0:
                    env = self.make_single_env(
                        env_id, 
                        seed=seed + idx if seed else None,
                        render_mode='rgb_array',
                        **kwargs
                    )
                    if video_folder:
                        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
                else:
                    env = self.make_single_env(
                        env_id,
                        seed=seed + idx if seed else None,
                        **kwargs
                    )
                
                # Apply standard wrappers
                env = gym.wrappers.RecordEpisodeStatistics(env)
                
                # Apply custom wrappers
                if wrappers:
                    for wrapper_cfg in wrappers:
                        env = instantiate(wrapper_cfg, env=env)
                
                return env
            
            return _thunk
        
        env_fns = [make_env(i) for i in range(num_envs)]
        
        VectorEnvClass = AsyncVectorEnv if self.async_envs else SyncVectorEnv
        return VectorEnvClass(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    
    def get_available_envs(self) -> List[str]:
        """Inference handles gym env lookup — no static list needed."""
        return []
    
    def validate_env_id(self, env_id: str) -> bool:
        """Check if environment ID exists in Gym registry."""
        return env_id in gym.envs.registry