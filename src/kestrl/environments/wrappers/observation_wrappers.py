# benchrl/environments/wrappers/observation_wrappers.py

import numpy as np
import gymnasium as gym
from collections import deque
from typing import Union, Tuple, Dict, Any

from .base_wrapper import ObservationWrapper


class NormalizeObservation(ObservationWrapper):
    """
    Normalize observations using running statistics.
    
    This wrapper will normalize observations s.t. each coordinate is centered with 
    zero mean and unit variance.
    """
    
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        
        # Running statistics
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float64)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float64)
        self.count = 0
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=-clip,
            high=clip,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Normalize the observation."""
        # Update running statistics
        self._update_stats(obs)
        
        # Normalize
        normalized_obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        normalized_obs = np.clip(normalized_obs, -self.clip, self.clip)
        
        return normalized_obs.astype(np.float32)
    
    def _update_stats(self, obs):
        """Update running mean and variance using Welford's algorithm."""
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        self.obs_var += delta * (obs - self.obs_mean) - self.obs_var / self.count


class RescaleObservation(ObservationWrapper):
    """
    Rescale observations to a new range.
    
    This wrapper rescales observations from the original range to a specified new range.
    """
    
    def __init__(self, env: gym.Env, new_min: float = 0.0, new_max: float = 1.0):
        super().__init__(env)
        self.new_min = new_min
        self.new_max = new_max
        self.old_min = env.observation_space.low.min()
        self.old_max = env.observation_space.high.max()
        
        # obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=new_min,
            high=new_max,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Rescale the observation."""
        
        # Rescale using linear transformation
        rescaled_obs = (obs - self.old_min) / (self.old_max - self.old_min) * (self.new_max - self.new_min) + self.new_min
        return rescaled_obs.astype(self.observation_space.dtype)

class FrameStackWrapper(ObservationWrapper):
    """
    Stack k last frames.
    
    Returns lazy array, which is much more memory efficient.
    """
    
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low, k, axis=0),
            high=np.repeat(env.observation_space.high, k, axis=0),
            shape=(k,) + obs_shape,
            dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
    def observation(self, obs):
        # Not used as we override reset and step
        return obs


class LazyFrames:
    """
    Lazy frame stacking to optimize memory usage.
    
    This object ensures that common frames between observations are only stored once.
    """
    
    def __init__(self, frames):
        self._frames = frames
        self._out = None
    
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]


class GrayScaleObservation(ObservationWrapper):
    """
    Convert RGB observations to grayscale.
    """
    
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        super().__init__(env)
        self.keep_dim = keep_dim
        
        obs_shape = self.observation_space.shape
        assert len(obs_shape) == 3 and obs_shape[-1] == 3, \
            "GrayScaleObservation only works with RGB images"
        
        if keep_dim:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=obs_shape[:-1] + (1,),
                dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=obs_shape[:-1],
                dtype=np.uint8
            )
    
    def observation(self, obs):
        """Convert observation to grayscale."""
        # RGB to grayscale formula: 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        gray = gray.astype(np.uint8)
        
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)
        
        return gray


class ResizeObservation(ObservationWrapper):
    """
    Resize image observations to a new shape.
    """
    
    def __init__(self, env: gym.Env, shape: Union[int, Tuple[int, int]]):
        super().__init__(env)
        
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=shape + obs_shape[2:],
            dtype=self.observation_space.dtype
        )
    
    def observation(self, obs):
        """Resize observation."""
        import cv2
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class FlattenObservation(ObservationWrapper):
    """
    Flatten observations.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)
    
    def observation(self, obs):
        """Flatten observation."""
        return gym.spaces.flatten(self.env.observation_space, obs)


class DtypeObservation(ObservationWrapper):
    """
    Convert observations to a specific dtype.
    """
    
    def __init__(self, env: gym.Env, dtype: np.dtype):
        super().__init__(env)
        self.dtype = dtype
        
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=dtype
        )
    
    def observation(self, obs):
        """Convert observation dtype."""
        return obs.astype(self.dtype)


class AddChannelDimension(ObservationWrapper):
    """
    Add a channel dimension to observations.
    
    Useful for converting 2D observations to 3D (H, W) -> (H, W, 1).
    """
    
    def __init__(self, env: gym.Env, axis: int = -1):
        super().__init__(env)
        self.axis = axis
        
        obs_shape = list(env.observation_space.shape)
        obs_shape.insert(axis, 1)
        
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=tuple(obs_shape),
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        """Add channel dimension."""
        return np.expand_dims(obs, axis=self.axis)


class PermuteObservation(ObservationWrapper):
    """
    Permute observation dimensions.
    
    Useful for converting between different channel conventions (CHW <-> HWC).
    """
    
    def __init__(self, env: gym.Env, permutation: Tuple[int, ...]):
        super().__init__(env)
        self.permutation = permutation
        
        obs_shape = env.observation_space.shape
        permuted_shape = tuple(obs_shape[i] for i in permutation)
        
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=permuted_shape,
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        """Permute observation dimensions."""
        return np.transpose(obs, self.permutation)


class ObservationDictToArray(ObservationWrapper):
    """
    Convert Dict observation space to array.
    
    Flattens and concatenates all values in the observation dictionary.
    """
    
    def __init__(self, env: gym.Env, keys_to_keep: Union[None, Tuple[str, ...]] = None):
        super().__init__(env)
        
        assert isinstance(env.observation_space, gym.spaces.Dict), \
            "ObservationDictToArray only works with Dict observation spaces"
        
        self.keys_to_keep = keys_to_keep
        if keys_to_keep is None:
            self.keys_to_keep = sorted(env.observation_space.spaces.keys())
        
        # Calculate flattened size
        flat_size = 0
        for key in self.keys_to_keep:
            space = env.observation_space[key]
            flat_size += gym.spaces.flatdim(space)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_size,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Convert dict observation to array."""
        obs_list = []
        for key in self.keys_to_keep:
            if key in obs:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    obs_list.append(val.flatten())
                else:
                    obs_list.append(np.array([val]))
        
        return np.concatenate(obs_list).astype(np.float32)


class SkipObservation(ObservationWrapper):
    """
    Skip observations by only returning every nth observation.
    
    Intermediate observations are discarded.
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer = obs
        return obs, info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        self._obs_buffer = obs
        return obs, total_reward, terminated, truncated, info
    
    def observation(self, obs):
        # Not used as we override reset and step
        return obs


# Utility function to chain multiple observation wrappers
def apply_observation_wrappers(env: gym.Env, wrappers_config: list) -> gym.Env:
    """
    Apply a chain of observation wrappers to an environment.
    
    Args:
        env: Base environment
        wrappers_config: List of wrapper configurations
            Each item should be a dict with '_target_' key and wrapper parameters
    
    Returns:
        Wrapped environment
    """
    from hydra.utils import instantiate
    
    for wrapper_cfg in wrappers_config:
        env = instantiate(wrapper_cfg, env=env)
    return env
