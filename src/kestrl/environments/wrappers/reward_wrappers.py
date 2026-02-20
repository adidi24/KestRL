import numpy as np
from .base_wrapper import BaseWrapper, RewardWrapper


class RewardScaleWrapper(RewardWrapper):
    """Scale rewards by a constant factor."""
    
    def __init__(self, env, scale: float = 1.0):
        self.scale = scale
        super().__init__(env)
    
    def reward(self, reward):
        return reward * self.scale


class RewardNoiseWrapper(RewardWrapper):
    """Add Gaussian noise to rewards."""
    
    def __init__(self, env, noise_std: float = 0.1):
        self.noise_std = noise_std
        super().__init__(env)
    
    def reward(self, reward):
        noise = np.random.normal(0, self.noise_std)
        return reward + noise


class ClipRewardWrapper(RewardWrapper):
    """Clip rewards to a range."""
    
    def __init__(self, env, min_reward: float = -1.0, max_reward: float = 1.0):
        self.min_reward = min_reward
        self.max_reward = max_reward
        super().__init__(env)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


class PositionDelayWrapper(BaseWrapper):
    """
    A Gymnasium wrapper that modifies the reward function based on position delay and control cost.
    This wrapper delays reward until the agent reaches a certain position (`position_delay`).
    It also penalizes large control signals to encourage smoother actions.

    Attributes:
        env (gym.Env): The environment to wrap.
        position_delay (float): Minimum x-position the agent must reach before receiving reward.
        ctrl_w (float): Weight for the control cost penalty term.
    """

    def __init__(
        self, env, position_delay: float = 2, ctrl_w: float = 0.001
    ) -> None:
        """
        Initialize the PositionDelayWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            position_delay (float): Minimum x-position the agent must reach before receiving reward.
            ctrl_w (float): Weight for the control cost penalty term.
        Returns:
            None
        """
        super().__init__(env)
        self.position_delay = position_delay
        self.ctrl_w = ctrl_w

    def step(self, action) -> tuple:
        """
        Take a step in the environment, modifying the reward.
        The environment's reward is replaced with a custom one that combines delayed
        forward movement reward and a control cost.

        Args:
            action: Action taken by the agent.
        Returns:
            tuple: (observation, modified_reward, terminated, truncated, info)
                - `info["x_pos"]`: Current x-position of the agent.
                - `info["action_norm"]`: Squared norm of the action.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["x_pos"] = self.unwrapped.data.qpos[0]
        info["action_norm"] = np.sum(np.square(action))
        return (
            observation,
            self.reward(action),
            terminated,
            truncated,
            info,
        )

    def reward(self, action: np.ndarray) -> float:
        """
        Compute the modified reward based on position delay and control penalty.

        Args:
            observation: Current observation (unused here, but kept for compatibility).
            action (np.ndarray): Action taken by the agent.
        Returns:
            float: Modified reward value.
        """
        x_pos = self.unwrapped.data.qpos[0]
        x_vel = self.unwrapped.data.qvel[0]
        ctrl_cost = self.ctrl_w * np.sum(np.square(action))
        forward_reward = (x_pos >= self.position_delay) * x_vel
        rewards = forward_reward - ctrl_cost
        return rewards
    
class PerturbRewardWrapper(RewardWrapper):
    """Perturb rewards by adding a small random value."""
    
    def __init__(self, env,
        e_=0.0,
        e=1.0,
        surrogate=False,
        epsilon=1e-6):
        
        assert (e_ + e <= 1.0)
        
        self.e_ = e_
        self.e = e
        self.surrogate = surrogate
        self.epsilon = epsilon
        super().__init__(env)

    def _noisy_reward(self, reward):
        n = np.random.random()
        if np.abs(reward - 1.0) < self.epsilon:
            if (n < self.e):
                return -1 * reward
        else:
            if (n < self.e_):
                return -1 * reward
        return reward

    def reward(self, reward):
        r = self._noisy_reward(reward)
        if not self.surrogate:
            return r

        if np.abs(r - 1.0) < self.epsilon:
            r_surrogate = ((1 - self.e_) * r + self.e * r) / (1 - self.e_ - self.e)
        else:
            r_surrogate = ((1 - self.e) * r + self.e_ * r) / (1 - self.e_ - self.e)
        
        print(f"Reward: {reward}, Noisy Reward: {r}, Surrogate Reward: {r_surrogate}")
        return r_surrogate
    


class InvertReward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        perturbed = -1 * reward
        return perturbed