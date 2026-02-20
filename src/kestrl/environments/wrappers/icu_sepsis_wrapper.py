import numpy as np
import gymnasium as gym


class ICUSepsisWrapper(gym.Wrapper):
    """Base wrapper class with common functionality."""
    
    # Environment constants
    _MAX_EPISODE_STEPS = 500

    # State meanings
    _STATE_DEATH = 713
    _STATE_SURVIVAL = 714
    _STATE_S_INF = 715
    _STATES_TERMINAL = {_STATE_DEATH, _STATE_SURVIVAL, _STATE_S_INF}
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._setup()
    
    def _setup(self):
        """Override this method to perform setup after initialization."""
        pass
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
        
    def _get_info(self) -> dict:
        info =  {
            'admissible_actions': self._admissible_actions[
                self._current_state],
            'state_vector': self._state_cluster_centers[self._current_state],
            'sofa_score': self._sofa_scores[self._current_state]}
        
        if self._current_state == self._STATE_DEATH:
            info['final_info'] = {'is_critical_failure': True}
        else:
            info['final_info'] = {'is_critical_failure': False}
        
        return info