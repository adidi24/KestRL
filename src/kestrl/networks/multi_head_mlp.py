"""MultiHead MLP — shared backbone with multiple named output heads.

Architecture:
    input → [shared hidden layers] → head_1 → output_1
                                   → head_2 → output_2
"""

from flax import nnx
import jax
import jax.numpy as jnp

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'tanh': jnp.tanh,
    'swish': jax.nn.swish,
}


class MultiHeadMLP(nnx.Module):
    """Multi-head MLP with shared backbone and parallel output heads."""

    def __init__(
        self,
        in_dim: int,
        head_configs: dict[str, int],
        hidden_dims: tuple[int, ...],
        activation: str = 'relu',
        head_activations: dict[str, str] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = ACTIVATIONS[activation]

        # Shared backbone
        self.backbone = nnx.List()
        self.backbone.append(nnx.Linear(in_dim, hidden_dims[0], rngs=rngs))
        for i in range(len(hidden_dims) - 1):
            self.backbone.append(nnx.Linear(hidden_dims[i], hidden_dims[i + 1], rngs=rngs))

        # Output heads
        self.heads = nnx.Dict({
            name: nnx.Linear(hidden_dims[-1], dim, rngs=rngs)
            for name, dim in head_configs.items()
        })

        self.head_activations = (
            {k: ACTIVATIONS[v] for k, v in head_activations.items()}
            if head_activations else {}
        )

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        for layer in self.backbone:
            x = self.activation(layer(x))

        return {
            name: self.head_activations.get(name, lambda x: x)(head(x))
            for name, head in self.heads.items()
        }
