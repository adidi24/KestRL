from typing import Any, NamedTuple
import warnings

import numpy as np
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

        # Store as float32 — halves data size vs float64 (MuJoCo default), reducing cache pressure
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype))
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available

            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )
            total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

        self._staging: np.ndarray | None = None   # allocated lazily on first sample call
        self._device = jax.devices()[0]

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
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = infos.get("TimeLimit.truncated", np.zeros(self.n_envs, dtype=bool))

        self.pos = (self.pos + 1) % self.buffer_size
        if not self.full and self.pos == 0:
            self.full = True

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        B = len(batch_inds)

        obs_dim = int(np.prod(self.obs_shape))
        act_dim = self.action_dim
        width = obs_dim * 2 + act_dim + 2          # obs | next_obs | acts | done | reward
        if self._staging is None or self._staging.shape[0] < B:
            self._staging = np.empty((B, width), dtype=np.float32)

        s = self._staging[:B]
        s[:, :obs_dim] = self.observations[batch_inds, env_indices].reshape(B, obs_dim)
        s[:, obs_dim:2*obs_dim] = self.next_observations[batch_inds, env_indices].reshape(B, obs_dim)
        s[:, 2*obs_dim:2*obs_dim+act_dim]    = self.actions[batch_inds, env_indices]
        s[:, 2*obs_dim+act_dim]              = (
            self.dones[batch_inds, env_indices]
            * (1 - self.timeouts[batch_inds, env_indices])
        )
        s[:, 2*obs_dim+act_dim+1]            = self.rewards[batch_inds, env_indices]

        gpu = jax.device_put(s, self._device)
        return ReplayBufferSamples(
            observations      = gpu[:, :obs_dim].reshape(B, *self.obs_shape),
            next_observations = gpu[:, obs_dim:2*obs_dim].reshape(B, *self.obs_shape),
            actions           = gpu[:, 2*obs_dim:2*obs_dim+act_dim],
            dones             = gpu[:, 2*obs_dim+act_dim:2*obs_dim+act_dim+1],
            rewards           = gpu[:, 2*obs_dim+act_dim+1:],
        )

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
