"""Brax environment factory and Gymnasium-compatible wrapper.

Environment IDs use the "brax/" prefix (e.g. "brax/ant", "brax/hopper").
"""

from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym

from ..base import EnvironmentFactory


class BraxVectorEnv:
    """Batched Brax env with a Gymnasium-compatible duck-type interface.

    Exposes:
      - numpy API (reset/step) for evaluation
      - JAX-native API (jax_reset/jax_step) for scan-based rollouts
    """

    def __init__(self, brax_env, num_envs: int, seed: Optional[int] = None) -> None:
        self._brax_env = brax_env
        self._num_envs = num_envs
        self._seed = seed if seed is not None else 0
        self._state = None

        self._jit_reset = jax.jit(self._brax_env.reset)
        self._jit_step  = jax.jit(self._brax_env.step)

        obs_size = self._brax_env.observation_size
        act_size = self._brax_env.action_size
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(act_size,),
            dtype=np.float32,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def brax_env(self):
        """Raw Brax env — JAX-traceable, safe to call inside lax.scan."""
        return self._brax_env

    @property
    def is_discrete(self) -> bool:
        return False

    # ── JAX-native API ─────────────────────────────────────────

    def jax_reset(self, key: jax.Array):
        return self._jit_reset(key)

    def jax_step(self, state, action: jax.Array):
        return self._jit_step(state, action)

    # ── Gymnasium-compatible API ────────────────────────────────

    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        key = jax.random.PRNGKey(seed if seed is not None else self._seed)
        self._state = self.jax_reset(key)
        return np.asarray(self._state.obs, dtype=np.float32), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)
        self._state = self.jax_step(self._state, actions_jax)
        obs         = np.asarray(self._state.obs,    dtype=np.float32)
        rewards     = np.asarray(self._state.reward, dtype=np.float32)
        done        = np.asarray(self._state.done,   dtype=bool)
        truncations = np.zeros_like(done, dtype=bool)
        return obs, rewards, done, truncations, {}

    def close(self) -> None:
        pass


class BraxEnvironmentFactory(EnvironmentFactory):
    """Factory for Brax physics environments.

    Env IDs must use the "brax/" prefix (e.g. "brax/ant").
    Brax is an optional dependency — ImportError raised at call time if not installed.
    """

    def make_single_env(
        self,
        env_id: str,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> BraxVectorEnv:
        import brax.envs
        brax_name = env_id.removeprefix("brax/")
        brax_env = brax.envs.create(brax_name, batch_size=1, **kwargs)
        return BraxVectorEnv(brax_env, num_envs=1, seed=seed)

    def make_vector_env(
        self,
        env_id: str,
        num_envs: int,
        seed: Optional[int] = None,
        capture_video: bool = False,
        video_folder: Optional[str] = None,
        wrappers: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> BraxVectorEnv:
        import brax.envs
        brax_name = env_id.removeprefix("brax/")
        brax_env = brax.envs.create(brax_name, batch_size=num_envs, **kwargs)
        return BraxVectorEnv(brax_env, num_envs=num_envs, seed=seed)

    def get_available_envs(self) -> List[str]:
        return []

    def validate_env_id(self, env_id: str) -> bool:
        try:
            import brax.envs
            brax_name = env_id.removeprefix("brax/")
            return brax_name in brax.envs.registered_envs
        except ImportError:
            return False
