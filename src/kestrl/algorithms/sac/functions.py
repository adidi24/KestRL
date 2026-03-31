"""Pure loss functions for SAC.

All functions are pure (no side effects), JIT-compatible.
They take network state + data → return (loss, metrics).

Key pattern:
    @nnx.jit
    def train_step(self):
        critic_loss, critic_metrics = compute_critic_loss(...)
        actor_loss, actor_metrics = compute_actor_loss(...)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from distrax import Categorical, Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -5
# ── Critic Loss ───────────────────────────────────────────────

def compute_discrete_critic_loss(
    critic: nnx.Module,
    # Target Q-values (precomputed by caller)
    target_q: jax.Array,
    obs: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, dict]:
    """Compute twin critic MSE loss.
    
    For discrete SAC: critic takes obs → Q(s, a) for ALL actions,
                      index with actions to get Q(s, a_taken)
    
    Args:
        critic: Critic network (outputs Q-values)
        target_q: Pre-computed target Q-values (batch,)
        obs: Observations (batch, obs_dim)
        actions: Actions taken (batch, action_dim)
    
    Returns:
        (loss, metrics_dict)
    """
    q_values = critic(obs)
    q_a_values = jnp.take_along_axis(q_values, actions.astype(jnp.int32), axis=1).reshape(-1)
    return jnp.mean((q_a_values - target_q) ** 2)

def compute_continuous_critic_loss(
    critic: nnx.Module,
    # Target Q-values (precomputed by caller)
    target_q: jax.Array,
    obs: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, dict]:
    """Compute twin critic MSE loss.

    For continuous SAC: critic takes concat(obs, action) → Q(s, a)
    
    Args:
        critic: Critic network (outputs Q-values)
        target_q: Pre-computed target Q-values (batch,)
        obs: Observations (batch, obs_dim)
        actions: Actions taken (batch, action_dim)
    
    Returns:
        (loss, metrics_dict)
    """
    critics_input = jnp.concatenate([obs, actions], axis=1)
    q_values = critic(critics_input).reshape(-1)
    return jnp.mean((q_values - target_q) ** 2)

# ── Actor Loss ────────────────────────────────────────────────

def compute_discrete_actor_loss(
    actor: nnx.Module,
    min_q_values: jax.Array,
    obs: jax.Array,
    alpha: float,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    """Actor loss for DISCRETE action spaces.

    Args:
        actor: Actor network (obs → action logits)
        min_q_values: Pre-computed target Q-values (batch,)
        obs: Observations (batch, obs_dim)
        alpha: Entropy coefficient
    
    Returns:
        (loss, metrics_dict) where metrics includes entropy
    """
    _, log_prob, action_probs = get_discrete_actor_action(actor, obs, key)
    actor_loss = (action_probs * ((alpha * log_prob) - min_q_values)).mean()
    return actor_loss, (log_prob, action_probs)


def compute_continuous_actor_loss(
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    obs: jax.Array,
    alpha: float,
    action_scale: jax.Array,
    action_bias: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    """Actor loss for CONTINUOUS action spaces.

    Args:
        actor: Actor network (obs → {mean, log_std})
        critic1, critic2: Twin critics
        obs: Observations (batch, obs_dim)
        alpha: Entropy coefficient
        key: PRNG key for sampling
    
    Returns:
        (loss, metrics_dict)
    """
    action, log_prob, _ = get_continuous_actor_action(actor, obs, action_scale, action_bias, key)
    critics_input = jnp.concatenate([obs, action], axis=1)
    qf1_values = critic1(critics_input)
    qf2_values = critic2(critics_input)
    min_q = jnp.minimum(qf1_values, qf2_values)
    actor_loss = jnp.mean((alpha * log_prob) - min_q)
    return actor_loss

# ── Get action ────────────────────────────────
def get_continuous_actor_action(
    actor: nnx.Module,
    obs: jax.Array,
    action_scale: jax.Array,
    action_bias: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    
    logits = actor(obs)
    mean = logits["mean"]
    log_std = jnp.clip(logits["log_std"], LOG_STD_MIN, LOG_STD_MAX) # important
    std = jnp.exp(log_std)
    normal = Normal(mean, std)
    x_t = normal.sample(seed=key)  # for reparameterization trick (mean + std * N(0,1))
    y_t = jnp.tanh(x_t)
    action = y_t * action_scale + action_bias
    log_prob = normal.log_prob(x_t)
    # Enforcing Action Bound
    log_prob -= jnp.log(action_scale * (1 - y_t ** 2) + 1e-6)
    log_prob = jnp.sum(log_prob, axis=1, keepdims=True)
    mean = jnp.tanh(mean) * action_scale + action_bias
    return action, log_prob, mean

def get_discrete_actor_action(
    actor: nnx.Module,
    obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    
    logits = actor(obs)
    policy_dist = Categorical(logits=logits)
    action = policy_dist.sample(seed=key)
    # Action probabilities for calculating the adapted soft-Q loss
    action_probs = policy_dist.probs
    log_prob = jax.nn.log_softmax(logits, axis=1)
    return action, log_prob, action_probs

# ── Get log probability ────────────────────────────────
def get_continuous_action_log_prob(
    actor: nnx.Module,
    obs: jax.Array,
    action: jax.Array,
    action_scale: jax.Array,
    action_bias: jax.Array,
) -> jax.Array:
    
    logits = actor(obs)
    mean = logits["mean"]
    log_std = jnp.clip(logits["log_std"], LOG_STD_MIN, LOG_STD_MAX) # important
    std = jnp.exp(log_std)
    normal = Normal(mean, std)
    
    unscaled_action = (action - action_bias) / action_scale
    unscaled_action = jnp.clip(unscaled_action, -1.0 + 1e-6, 1.0 - 1e-6)
    raw_action = jnp.atanh(unscaled_action)
    
    log_prob = normal.log_prob(raw_action)
    log_prob -= jnp.log(action_scale * (1 - unscaled_action ** 2) + 1e-6)
    log_prob = jnp.sum(log_prob, axis=1)
    return log_prob

def get_discrete_action_log_prob(
    actor: nnx.Module,
    obs: jax.Array,
    action: jax.Array,
) -> jax.Array:
    
    logits = actor(obs)
    policy_dist = Categorical(logits=logits)
    log_prob = policy_dist.log_prob(action)
    return log_prob
