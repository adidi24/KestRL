from typing import Any, NamedTuple
import warnings

import numpy as np
import jax.numpy as jnp
import jax

from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class RolloutBufferSamples(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    old_values: jax.Array
    old_log_prob: jax.Array
    advantages: jax.Array
    returns: jax.Array


class ReplayBufferSamples(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    rewards: jax.Array

class ReplayBuffer:
    """
    Replay buffer class

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        handle_timeout_termination: bool = True
    ):
        super().__init__()
        self.buffer_size = max(buffer_size // n_envs, 1)
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.obs_shape = observation_space.shape
        if hasattr(self.action_space, 'n'):
            # For discrete spaces
            self.action_dim = 1
        else:
            self.action_dim = int(np.prod(self.action_space.shape))
            
        
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype))
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
        
        
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        
        Args:
            arr: (np.ndarray)
        
        Returns:
            np.ndarray
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        Returns:
            Current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: dict[str, Any]) -> None:
        """
        Add elements to the buffer.
        """
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                infos.get("TimeLimit.truncated", np.zeros(self.n_envs, dtype=bool))
            )


        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        Args:
            batch_size (int) : Number of element to sample
            env : associated gym VecEnv
                to normalize the observations/rewards when sampling
        
        Returns:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)
    
    def _get_samples(self, batch_inds: np.ndarray) -> tuple:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_jnp, data)))

    def to_jnp(self, array: np.ndarray, dtype: jnp.dtype = jnp.float32) -> jnp.array:
        """
        Convert a numpy array to a jax numpy array.

        Args:
            array (np.ndarray): Array to convert
        
        Returns:
            jnp.array
        """
        return jnp.array(array, dtype=dtype)
    
    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype