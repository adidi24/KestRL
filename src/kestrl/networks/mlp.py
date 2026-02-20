"""Multi-Layer Perceptron — thin wrapper around MultiHeadMLP with a single output head."""

from flax import nnx
from kestrl.networks.multi_head_mlp import MultiHeadMLP


class MLP(nnx.Module):
    """Single-output MLP. Delegates to MultiHeadMLP internally."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: tuple[int, ...],
                 activation: str = 'relu', *, rngs: nnx.Rngs):
        self._net = MultiHeadMLP(in_dim, {'out': out_dim}, hidden_dims,
                                 activation, rngs=rngs)

    def __call__(self, x):
        return self._net(x)['out']
