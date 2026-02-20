from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
import gymnasium as gym
from gymnasium.vector import VectorEnv

class EnvironmentFactory(ABC):
    """Abstract base class for environment factories."""
    
    @abstractmethod
    def make_single_env(
        self,
        env_id: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ) -> gym.Env:
        """Create a single environment instance."""
        pass
    
    @abstractmethod
    def make_vector_env(
        self,
        env_id: str,
        num_envs: int,
        seed: Optional[int] = None,
        capture_video: bool = False,
        video_folder: Optional[str] = None,
        wrappers: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> VectorEnv:
        """Create a vectorized environment."""
        pass
    
    @abstractmethod
    def get_available_envs(self) -> List[str]:
        """Return list of available environment IDs."""
        pass
    
    @abstractmethod
    def validate_env_id(self, env_id: str) -> bool:
        """Check if environment ID is valid for this factory."""
        pass


class EnvironmentBuilder:
    """Unified interface for building environments from any API."""
    
    def __init__(self):
        self._factories: Dict[str, EnvironmentFactory] = {}
        self._env_mappings: Dict[str, str] = {}  # env_id -> factory_name
    
    def register_factory(self, name: str, factory: EnvironmentFactory):
        """Register an environment factory."""
        self._factories[name] = factory
        
        # Update environment mappings
        for env_id in factory.get_available_envs():
            self._env_mappings[env_id] = name
    
    def build_env(
        self,
        env_id: str,
        num_envs: int = 1,
        seed: Optional[int] = None,
        capture_video: bool = False,
        video_folder: Optional[str] = None,
        wrappers: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Union[gym.Env, VectorEnv]:
        """Build environment using appropriate factory."""
        
        # Find appropriate factory
        factory_name = self._env_mappings.get(env_id)
        if factory_name is None:
            # Try to infer factory from env_id pattern
            factory_name = self._infer_factory(env_id)
        
        if factory_name not in self._factories:
            raise ValueError(f"No factory found for environment '{env_id}'")
        
        factory = self._factories[factory_name]
        
        return factory.make_vector_env(
            env_id=env_id,
            num_envs=num_envs,
            seed=seed,
            capture_video=capture_video,
            video_folder=video_folder,
            wrappers=wrappers,
            **kwargs
        )
    
    def _infer_factory(self, env_id: str) -> str:
        """Infer factory from environment ID pattern."""
        if env_id.startswith("ALE/"):
            return "gym"
        elif env_id.startswith("mt-"):
            return "metaworld"
        elif env_id.startswith("dm_control/"):
            return "dm_control"
        else:
            return "gym"  # Default to gym
    
    def _apply_wrappers(self, env: gym.Env, wrappers: List[Dict[str, Any]]) -> gym.Env:
        """Apply wrappers to environment."""
        from hydra.utils import instantiate
        
        # breakpoint()
        for wrapper_cfg in wrappers:
            env = instantiate(wrapper_cfg, env=env)
        return env