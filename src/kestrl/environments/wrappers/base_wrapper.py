import gymnasium as gym
from typing import Any, Dict, Optional


class BaseWrapper(gym.Wrapper):
    """Base wrapper class with common functionality."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._setup()
    
    def _setup(self):
        """Override this method to perform setup after initialization."""
        pass
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step through the environment."""
        return self.env.step(action)
    
    @property                                                                                       
    def is_discrete(self) -> bool:                          
        return isinstance(self.single_action_space, gym.spaces.Discrete)  


class ObservationWrapper(BaseWrapper):
    """Base class for observation wrappers."""
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
    
    def observation(self, obs):
        """Transform the observation. Override this."""
        raise NotImplementedError


class RewardWrapper(BaseWrapper):
    """Base class for reward wrappers."""
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward(reward), terminated, truncated, info
    
    def reward(self, reward):
        """Transform the reward. Override this."""
        raise NotImplementedError


class ActionWrapper(BaseWrapper):
    """Base class for action wrappers."""
    
    def step(self, action):
        return self.env.step(self.action(action))
    
    def action(self, action):
        """Transform the action. Override this."""
        raise NotImplementedError