"""Pure functions for PBSAC.

All functions are pure (no side effects), JIT-compatible.
They take network state + data → return (loss, metrics).

"""

import jax
import jax.numpy as jnp
from flax import nnx
from kestrl.algorithms.functions.sac_losses import (
    get_discrete_action_log_prob,
    get_continuous_action_log_prob,
    get_discrete_actor_action,
    get_continuous_actor_action,
)
from kestrl.distributions import (
    _construct_state_from_flat_state,
    block_sample,
    kl_block,
    BlockPosterior,
    BlockPrior
)

def compute_policy_is_return(
    actor: nnx.Module,
    is_discrete: bool,
    batch_data: dict,
) -> jnp.ndarray:
    """Computes the importance-sampled empirical return."""
    def get_traj_log_prob_ratio(states, actions, log_probs_b, mask):
        if is_discrete:
            log_probs = get_discrete_action_log_prob(actor, states, actions)
        else:
            log_probs = get_continuous_action_log_prob(
                actor, states, actions, 
                batch_data['action_scale'], batch_data['action_bias']
            )
        
        log_ratios = log_probs - log_probs_b
        
        # Mask out padding steps
        log_ratios = jnp.where(mask, log_ratios, 0.0)

        return jnp.sum(log_ratios)

    batch_log_probs_ratio = jax.vmap(get_traj_log_prob_ratio)(
        batch_data['states'], 
        batch_data['actions'], 
        batch_data['log_probs_b'], 
        batch_data['masks']
    )
    

    # We standardize the batch_log_probs_ratio so they have a mean of 0 and std of 1.
    rho_std = jnp.std(batch_log_probs_ratio) + 1e-8
    rho_mean = jnp.mean(batch_log_probs_ratio)
    tau = 0.5
    scaled_log_probs_ratio = (batch_log_probs_ratio - rho_mean) / (rho_std * tau)
    
    weights = jax.nn.softmax(scaled_log_probs_ratio)
    estimated_return = jnp.sum(weights * batch_data['returns'])
    
    return estimated_return

def compute_pac_bayes_loss(
    posterior: BlockPosterior,
    prior: BlockPrior,
    actor: nnx.Module,  # Pass this so we can inject params
    is_discrete: bool,
    key: jax.Array,
    num_samples: int,
    lambda_val: float,
    C_const: float,
    C_prime_const: float,
    batch_data: dict,
) -> tuple[jnp.ndarray, dict]:
    """Computes the PAC-Bayes loss."""
    
    estimated_returns = []
    current_key = key
    
    for _ in range(num_samples):
        current_key, sub_key = jax.random.split(current_key)
        flat_state = block_sample(posterior, sub_key)
        
        actor_state = _construct_state_from_flat_state(actor, flat_state, posterior.shapes)
        graphdef, _ = nnx.split(actor)
        temp_actor = nnx.merge(graphdef, actor_state)
        
        ret = compute_policy_is_return(
            temp_actor, is_discrete, batch_data
        )
        estimated_returns.append(ret)

    estimated_returns = jnp.stack(estimated_returns)
    mean_return = jnp.mean(estimated_returns)

    kl   = kl_block(posterior, prior)
    
    L_bound = (C_const * kl + C_prime_const) / (2.0 * lambda_val) + (lambda_val / 2.0)
    loss = -mean_return + L_bound
    
    return loss, {
        'loss': loss,
        'kl_div': kl,
        'mean_empirical_return': mean_return,
        'lambda': lambda_val
    }

@nnx.jit(static_argnames=('is_discrete', 'num_samples'))
def compute_pac_bayes_bound(
    posterior: BlockPosterior,
    prior: BlockPrior,
    actor: nnx.Module,
    is_discrete: bool,
    key: jax.Array,
    num_samples: int,
    T: int,
    H: int,
    r_max: float,
    mixing_time: int,
    gamma: float,
    delta: float,
    batch_data: dict,
) -> dict:
    """Computes the PAC-Bayes bound on test trajectories."""
    
    estimated_returns = []
    current_key = key
    
    for _ in range(num_samples):
        current_key, sub_key = jax.random.split(current_key)
        flat_state = block_sample(posterior, sub_key)
        
        actor_state = _construct_state_from_flat_state(actor, flat_state, posterior.shapes)
        graphdef, _ = nnx.split(actor)
        temp_actor = nnx.merge(graphdef, actor_state)
        
        ret = compute_policy_is_return(
            temp_actor, is_discrete, batch_data
        )
        estimated_returns.append(ret)

    estimated_returns = jnp.stack(estimated_returns)
    mean_return = jnp.mean(estimated_returns)

    kl = kl_block(posterior, prior)
    
    c_squared_numerator = r_max ** 2 * (1 - gamma ** (2 * H))
    c_squared_denominator = T * (1 - gamma ** 2)
    c_squared = jnp.maximum(1e-6, c_squared_numerator / c_squared_denominator)
    
    term_inside_sqrt = c_squared * mixing_time * (kl + jnp.log(jnp.sqrt(2.0) / delta))
    uncertainty_term = jnp.sqrt(term_inside_sqrt)
    
    certified_return = mean_return - uncertainty_term
    
    return {
        'certified_return': certified_return,
        'empirical_return': mean_return,
        'uncertainty_term': uncertainty_term,
        'kl_div': kl,
        'c_squared': c_squared,
        'mixing_time': mixing_time,
        'r_max': r_max,
    }

def compute_posterior_guided_targets(
    posterior: BlockPosterior,
    actor: nnx.Module,
    target_critic1: nnx.Module,
    target_critic2: nnx.Module,
    log_alpha: nnx.Module,
    next_obs: jnp.ndarray,    # (B, obs_dim)
    rewards: jnp.ndarray,     # (B,) or (B, 1)
    dones: jnp.ndarray,       # (B,) or (B, 1)
    gamma: float,
    n_samples: int,
    key: jax.Array,
    is_discrete: bool,
    # Continuous only:
    action_scale: jnp.ndarray | None = None,
    action_bias: jnp.ndarray | None = None,
) -> jnp.ndarray:             # (B,) — averaged target_q
    """Average TD targets over n_samples policies drawn from the posterior.
    
    This is used during the critic adaptation phase when the actor is frozen."""
    B = next_obs.shape[0]
    alpha = jnp.exp(log_alpha.value[...])
    target_q_accumulator = jnp.zeros(B,)
    
    for i in range(n_samples):
        if i == 0:
            flat_state = {name: lp.mean.get_value()
                          for name, lp in posterior.layers.items()}
        else:
            key, subkey = jax.random.split(key)
            flat_state = block_sample(posterior, subkey)
        
        actor_state = _construct_state_from_flat_state(actor, flat_state, posterior.shapes)
        graphdef, _ = nnx.split(actor)
        temp_actor = nnx.merge(graphdef, actor_state)
        
        # Get next action + log_prob from this sample
        key, act_key = jax.random.split(key)
        if is_discrete:
            _, next_log_pi, next_action_probs = get_discrete_actor_action(temp_actor, next_obs, act_key)
            min_next_q = jnp.minimum(
                target_critic1(next_obs), target_critic2(next_obs)
            ) - alpha * next_log_pi
            min_next_q = jnp.sum(next_action_probs * min_next_q, axis=1)
            target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * min_next_q
        else:
            next_action, next_log_pi, _ = get_continuous_actor_action(
                temp_actor, next_obs, action_scale, action_bias, act_key
            )
            critics_input = jnp.concatenate([next_obs, next_action], axis=1)
            min_next_q = jnp.minimum(
                target_critic1(critics_input),
                target_critic2(critics_input),
            ) - alpha * next_log_pi
            target_q = rewards.flatten() + (1 - dones.flatten()) * gamma * min_next_q.reshape(-1)
        
        target_q_accumulator += target_q
    
    return target_q_accumulator / n_samples

# ── Get action ────────────────────────────────
def get_continuous_actor_action_from_posterior(
    posterior: BlockPosterior,
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    obs: jax.Array,
    action_scale: jax.Array,
    action_bias: jax.Array,
    explore_prob: float,
    explore_n_samples: int,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    
    if jax.random.uniform(key) < explore_prob:
        candidate_actions = []
        candidate_log_probs = []
        candidate_means = []
        candidate_q_values = []
        
        for i in range(explore_n_samples):
            key, sub_key, act_key = jax.random.split(key, 3)
            if i == 0:
                flat_state = {name: lp.mean.get_value()
                            for name, lp in posterior.layers.items()}
            else:
                flat_state = block_sample(posterior, sub_key)
            
            actor_state = _construct_state_from_flat_state(actor, flat_state, posterior.shapes)
            graphdef, _ = nnx.split(actor)
            temp_actor = nnx.merge(graphdef, actor_state)
            
            action, log_prob, mean = get_continuous_actor_action(
                temp_actor, obs, action_scale, action_bias, act_key
            )
            
            critics_input = jnp.concatenate([obs, action], axis=1)
            q = jnp.minimum(critic1(critics_input), critic2(critics_input)).reshape(-1)
            candidate_actions.append(action)
            candidate_log_probs.append(log_prob)
            candidate_means.append(mean)
            candidate_q_values.append(q)
        
        q_values = jnp.stack(candidate_q_values)
        best_idx = jnp.argmax(q_values, axis=0)
        candidate_actions = jnp.stack(candidate_actions)
        candidate_log_probs = jnp.stack(candidate_log_probs)
        candidate_means = jnp.stack(candidate_means)
        
        action = candidate_actions[best_idx, jnp.arange(best_idx.shape[0])]
        log_prob = candidate_log_probs[best_idx, jnp.arange(best_idx.shape[0])]
        mean = candidate_means[best_idx, jnp.arange(best_idx.shape[0])]
    else:
        action, log_prob, mean = get_continuous_actor_action(
              actor, obs, action_scale, action_bias, key
          )

    return action, log_prob, mean

def get_discrete_actor_action_from_posterior(
    posterior: BlockPosterior,
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    obs: jax.Array,
    explore_prob: float,
    explore_n_samples: int,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    
    if jax.random.uniform(key) < explore_prob:
        candidate_actions = []
        candidate_log_probs = []
        candidate_action_probs = []
        candidate_q_values = []
        
        for i in range(explore_n_samples):
            key, sub_key, act_key = jax.random.split(key, 3)
            if i == 0:
                flat_state = {name: lp.mean.get_value()
                            for name, lp in posterior.layers.items()}
            else:
                flat_state = block_sample(posterior, sub_key)
            
            actor_state = _construct_state_from_flat_state(actor, flat_state, posterior.shapes)
            graphdef, _ = nnx.split(actor)
            temp_actor = nnx.merge(graphdef, actor_state)
            
            action, log_prob, action_probs = get_discrete_actor_action(
                temp_actor, obs, act_key
            )
            
            q_all = jnp.minimum(critic1(obs), critic2(obs))                                                                                                             
            q = jnp.take_along_axis(q_all, action.reshape(-1, 1).astype(jnp.int32), axis=1).reshape(-1)  
            candidate_actions.append(action)
            candidate_log_probs.append(log_prob)
            candidate_action_probs.append(action_probs)
            candidate_q_values.append(q)
        
        q_values = jnp.stack(candidate_q_values)
        best_idx = jnp.argmax(q_values, axis=0)
        candidate_actions = jnp.stack(candidate_actions)
        candidate_log_probs = jnp.stack(candidate_log_probs)
        candidate_action_probs = jnp.stack(candidate_action_probs)
        
        action = candidate_actions[best_idx, jnp.arange(best_idx.shape[0])]
        log_prob = candidate_log_probs[best_idx, jnp.arange(best_idx.shape[0])]
        action_probs = candidate_action_probs[best_idx, jnp.arange(best_idx.shape[0])]
    else:
        action, log_prob, action_probs = get_discrete_actor_action(
              actor, obs, key
          )
    return action, log_prob, action_probs