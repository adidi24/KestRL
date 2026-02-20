import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from benchrl.environments.wrappers.observation_wrappers import RescaleObservation

def AtariPreprocessing(env:gym.Env, frame_stack: int = 4, frame_skip: int = 4):
    """Create and preprocess gym environment with standard wrappers."""
    # Create Atari environment with wrappers 

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env = RescaleObservation(env, new_min=0.0, new_max=1.0)
    return env
