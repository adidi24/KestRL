from kestrl.buffers.jax_replay_buffer import JAXTransition, JAXBufferState, _empty_buffer, _buffer_add, _buffer_sample
from kestrl.buffers.replay_buffer import ReplayBuffer

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
    
__all__ = ['JAXTransition',
           'JAXBufferState',
           '_empty_buffer',
           '_buffer_add',
           '_buffer_sample',
           'RolloutBufferSamples',
           'ReplayBufferSamples',
           'ReplayBuffer']