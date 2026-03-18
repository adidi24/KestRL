"""JIT-compiled update functions for PBSAC.

Pure functions that perform gradient computation + optimizer steps.
Each function takes all required nnx.Module objects as explicit arguments,
making them compatible with @nnx.jit without requiring the SAC class to
be an nnx.Module itself.

"""

import jax
import jax.numpy as jnp
from flax import nnx
from kestrl.algorithms.functions.pbsac_losses import compute_pac_bayes_loss
from kestrl.algorithms.functions.sac_losses import (
    compute_discrete_critic_loss,
    compute_continuous_critic_loss,
)

@nnx.jit(static_argnames=('is_discrete', 'num_samples'))
def update_pb_posterior(
    posterior: nnx.Module,
    prior: nnx.Module,
    actor: nnx.Module,
    optimizer: nnx.Optimizer,
    is_discrete: bool,
    key: jax.Array,
    num_samples: int,
    lambda_val: float,
    C_const: float,
    C_prime_const: float,
    batch_data: dict,
):
    loss_fn = lambda p: compute_pac_bayes_loss(p, prior, actor,
                                               is_discrete,
                                               key,
                                               num_samples,
                                               lambda_val,
                                               C_const,
                                               C_prime_const,
                                               batch_data)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(posterior)
    optimizer.update(posterior, grads)
    return metrics

# ── Critic Discrete ────────────────────────────────────────────

@nnx.jit
def adaptive_update_discrete_critics(
    critic1: nnx.Module,
    critic2: nnx.Module,
    critic1_opt: nnx.Optimizer,
    critic2_opt: nnx.Optimizer,
    obs: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
) -> dict[str, jax.Array]:
    """Critic-only update for discrete action spaces during the actor-frozen phase.

    target_q is precomputed by averaging over posterior samples — actor and alpha
    are not updated here since the actor is frozen.
    """
    # Update critic1
    def c1_loss(c1):
        return compute_discrete_critic_loss(c1, target_q, obs, actions)
    q1_loss, grads1 = nnx.value_and_grad(c1_loss)(critic1)
    critic1_opt.update(critic1, grads1)

    # Update critic2
    def c2_loss(c2):
        return compute_discrete_critic_loss(c2, target_q, obs, actions)
    q2_loss, grads2 = nnx.value_and_grad(c2_loss)(critic2)
    critic2_opt.update(critic2, grads2)

    return {'critic/loss': q1_loss + q2_loss}

# ── Critic Updates ────────────────────────────────────────────
@nnx.jit
def adaptive_update_continuous_critics(
    critic1: nnx.Module,
    critic2: nnx.Module,
    critic1_opt: nnx.Optimizer,
    critic2_opt: nnx.Optimizer,
    obs: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
) -> dict[str, jax.Array]:
    """Update twin critics for continuous action spaces."""

    # Critic 1
    def c1_loss(c1):
        return compute_continuous_critic_loss(c1, target_q, obs, actions)
    q1_loss, grads1 = nnx.value_and_grad(c1_loss)(critic1)
    critic1_opt.update(critic1, grads1)

    # Critic 2
    def c2_loss(c2):
        return compute_continuous_critic_loss(c2, target_q, obs, actions)
    q2_loss, grads2 = nnx.value_and_grad(c2_loss)(critic2)
    critic2_opt.update(critic2, grads2)

    return {'critic/loss': q1_loss + q2_loss}