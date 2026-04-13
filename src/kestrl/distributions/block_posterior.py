from functools import reduce

import jax
import jax.numpy as jnp
from flax import nnx


class LayerPosterior(nnx.Module):
    """Per-layer posterior N(mean, P Pᵀ + diag(std²))."""

    def __init__(self, mean: jnp.ndarray, std: jnp.ndarray, P: jnp.ndarray):
        self.mean = nnx.Param(mean)   # (D_l,) flattened mean
        safe_std = jnp.clip(std, 1e-8, None)
        pre_std = jnp.log(jnp.exp(safe_std) - 1.0)
        self.pre_std  = nnx.Param(pre_std)    # (D_l,) diagonal pre_std
        self.P    = nnx.Param(P)      # (D_l, K) low-rank factor — zeros at init

    @property
    def std(self):
        return jax.nn.softplus(self.pre_std.get_value())

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Sample w ~ N(mean, P Pᵀ + diag(std²)) via split reparameterisation."""
        key_diag, key_lr = jax.random.split(key)
        m, s, p = self.mean.get_value(), self.std, self.P.get_value()
        eps_diag = jax.random.normal(key_diag, m.shape)
        eps_lr   = jax.random.normal(key_lr,   (p.shape[1],))
        return m + s * eps_diag + p @ eps_lr

class BlockPosterior(nnx.Module):
    """Block-wise posterior: one LayerPosterior per network parameter tensor."""
    layers: dict = nnx.data()   # nnx.Module inherits Pytree rules — dict needs nnx.data()

    def __init__(self, layers: dict, rank: int, shapes: dict | None = None):
        self.layers = layers
        self.rank   = rank     # plain int — not a parameter
        self.shapes = shapes

    @staticmethod
    def from_actor(
        actor: nnx.Module,
        rank: int = 10,
        init_std: float = 0.01,
        fixed_layers_depth: int = 0,
    ) -> 'BlockPosterior':
        """Build a BlockPosterior from a Flax NNX actor.

        mean  = current actor weights (flattened per tensor)
        std   = init_std (uniform; Xavier-aware init is a future option)
        P     = zeros → at init the posterior is purely diagonal (KL = 0)

        Args:
            actor: Flax NNX actor module.
            rank: Low-rank dimension for covariance factorisation.
            init_std: Initial posterior standard deviation.
            fixed_layers_depth: Number of leading modules to keep deterministic.
                When > 0, the first N unique parameter modules identified during
                state flattening are excluded from the posterior.
        """
        params_tree = nnx.state(actor, nnx.Param)
        flat_state = list(nnx.to_flat_state(params_tree))
        
        # Discover unique layer modules in the order they appear
        # in the model state (top-to-bottom for standard models)
        unique_modules = []
        for path, _ in flat_state:
            mod_path = path[:-1]
            if mod_path not in unique_modules:
                unique_modules.append(mod_path)
        
        frozen_modules = set(unique_modules[:fixed_layers_depth])

        layers = {}
        shapes = {}
        for path, param in flat_state:
            # Optionally skip leading layers based on depth
            if fixed_layers_depth > 0 and path[:-1] in frozen_modules:
                continue

            mean = param.get_value().flatten()
            std  = jnp.full_like(mean, init_std)
            P    = jnp.zeros((mean.shape[0], rank))
            layers[path] = LayerPosterior(mean=mean, std=std, P=P)
            shapes[path] = param.get_value().shape
        return BlockPosterior(layers=layers, rank=rank, shapes=shapes)

class BlockPrior(nnx.Module):
    """Block-wise prior: diagonal only (no P term) per layer.

    Stores plain arrays (not nnx.Param) — the prior is not optimised,
    only updated via EMA from the outside.
    """
    layers: dict = nnx.data()   # dict of plain-array tuples — still needs nnx.data()

    def __init__(self, layers: dict):
        self.layers = layers

    @staticmethod
    def from_posterior(posterior: BlockPosterior) -> 'BlockPrior':
        """Initialise prior as a copy of the posterior (KL = 0 at start)."""
        return BlockPrior(
            layers={name: (lp.mean.get_value().copy(), lp.std.copy())
                    for name, lp in posterior.layers.items()}
        )


# ---------------------------------------------------------------------------
# Functional API — pure functions, JIT-able
# ---------------------------------------------------------------------------

def block_sample(posterior: BlockPosterior, key: jax.Array) -> dict:
    """Sample one policy parameter dict from the posterior.

    Returns {param_path: flat_weight_array}. The caller reshapes each array
    back to the original tensor shape before loading into the actor.
    """
    sampled = {}
    for name, lp in posterior.layers.items():
        key, k1, k2 = jax.random.split(key, 3)
        m, s, p = lp.mean.get_value(), lp.std, lp.P.get_value()
        eps_diag = jax.random.normal(k1, m.shape)
        eps_lr   = jax.random.normal(k2, (posterior.rank,))
        sampled[name] = m + s * eps_diag + p @ eps_lr
    return sampled

def ema_update_prior(prior: BlockPrior, posterior: BlockPosterior, decay: float = 0.99) -> 'BlockPrior':
        """Slide prior toward posterior: μ₀ ← ι·υ + (1-ι)·μ₀ (Algorithm 1)."""
        prior.layers = {
            name: (
                decay * posterior.layers[name].mean.get_value() + (1 - decay) * m0,
                decay * posterior.layers[name].std  + (1 - decay) * s0,
            )
            for name, (m0, s0) in prior.layers.items()
        }
        return prior

def _construct_state_from_flat_state(
    sampled_flat_state: dict,
    shapes: dict,
    base_state: dict | None = None,
) -> dict:
    """Build an actor state by overlaying sampled params onto a base state.

    When the posterior only covers a subset of actor layers (via
    ``fixed_layers_depth``), ``base_state`` must be supplied.  It provides
    the full set of actor parameters — the sampled posterior values are
    then written on top of the matching paths while backbone layers keep
    their deterministic values.

    Args:
        sampled_flat_state: ``{path: flat_array}`` from ``block_sample``.
        shapes: ``{path: shape}`` from ``posterior.shapes``.
        base_state: If not ``None``, a flat ``{path: array}`` dict of the
            current actor params (all layers).  Paths present in
            ``sampled_flat_state`` are overwritten; the rest pass through.

    Returns:
        A nested ``nnx.State`` ready for ``nnx.merge(actor_gd, state)``.
    """
    if base_state is not None:
        # Start from the full actor state and overwrite posterior layers
        new_flat = {path: val for path, val in base_state.items()}
        for path in sampled_flat_state:
            new_flat[path] = sampled_flat_state[path].reshape(shapes[path])
    else:
        new_flat = {
            path: sampled_flat_state[path].reshape(shapes[path])
            for path in sampled_flat_state
        }
    state = nnx.from_flat_state(new_flat)
    return state
        
def _layer_kl(lp: LayerPosterior, prior_layer: tuple) -> jnp.ndarray:
    """ Uses the Matrix Determinant Lemma for log det Σ and Woodbury for the
    trace term — both O(D·K + K³) instead of O(D³).
    """
    mean_q = lp.mean.get_value()
    std_q  = lp.std
    P_q    = lp.P.get_value()
    mean_p, std_p = prior_layer
    D = mean_q.shape[0]

    # --- Log-det term ---
    inv_var_q = std_q ** -2
    M = jnp.eye(P_q.shape[1]) + P_q.T @ (inv_var_q[:, None] * P_q)  # (K, K)
    _, logdet_M = jnp.linalg.slogdet(M)
    log_det_term = (
        jnp.sum(jnp.log(std_p ** 2))
        - jnp.sum(jnp.log(std_q ** 2))
        - logdet_M
    )

    # --- Trace term ---
    inv_var_p = std_p ** -2
    trace_core = jnp.trace(P_q.T @ (inv_var_p[:, None] * P_q))
    trace_term = jnp.sum(std_q ** 2 * inv_var_p) + trace_core

    # --- Quadratic + constant ---
    quad_term = jnp.sum((mean_q - mean_p) ** 2 * inv_var_p)

    return 0.5 * (log_det_term - D + quad_term + trace_term)


def kl_block(posterior: BlockPosterior, prior: BlockPrior) -> jnp.ndarray:
    """Total KL(posterior || prior) = sum of per-layer KLs."""
    return sum(_layer_kl(lp, prior.layers[name])
               for name, lp in posterior.layers.items())