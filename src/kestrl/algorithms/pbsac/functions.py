"""Pure functions for PBSAC.

All functions are pure (no side effects), JIT-compatible.
They take network state + data → return (loss, metrics).

"""

import jax
import jax.numpy as jnp
from flax import nnx
from kestrl.algorithms.sac.functions import (
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
    
    graphdef, _ = nnx.split(actor)
    for _ in range(num_samples):
        current_key, sub_key = jax.random.split(current_key)
        sampled_flat_state = block_sample(posterior, sub_key)
        
        sampled_actor_state = _construct_state_from_flat_state(sampled_flat_state, posterior.shapes)
        temp_actor = nnx.merge(graphdef, sampled_actor_state)
        
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
    
    graphdef, _ = nnx.split(actor)
    for _ in range(num_samples):
        current_key, sub_key = jax.random.split(current_key)
        flat_state = block_sample(posterior, sub_key)
        
        actor_state = _construct_state_from_flat_state(flat_state, posterior.shapes)
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

# ── Get action ────────────────────────────────

def get_actor_discrete_action_from_posterior_vmap(
    posterior: BlockPosterior,
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    obs: jax.Array,
    should_explore: bool,
    explore_n_samples: int,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    
    if should_explore:
        graphdef, _ = nnx.split(actor)
        
        def sample_and_apply(key):
            sub_key1, sub_key2 = jax.random.split(key)
            flat_state = block_sample(posterior, sub_key1)
            actor_state = _construct_state_from_flat_state(flat_state, posterior.shapes)
            temp_actor = nnx.merge(graphdef, actor_state)

            action, log_prob, action_probs = get_discrete_actor_action(temp_actor, obs, sub_key2)
            q_values = jnp.minimum(
                critic1(obs), critic2(obs)
            )
            q_value = jnp.take_along_axis(q_values, action.reshape(-1, 1).astype(jnp.int32), axis=1).reshape(-1)  
            return {'q_values': q_value, 'action': action, 'log_prob': log_prob, 'aux': action_probs}
            
        keys = jax.random.split(key, explore_n_samples)
        results = jax.vmap(sample_and_apply)(keys)
        best_idx = jnp.argmax(results['q_values'], axis=0)
        
        action = results['action'][best_idx, jnp.arange(best_idx.shape[0])]
        log_prob = results['log_prob'][best_idx, jnp.arange(best_idx.shape[0])]
        aux = results['aux'][best_idx, jnp.arange(best_idx.shape[0])]
    else:
        action, log_prob, aux = get_discrete_actor_action(
            actor, obs, key
        )

    return action, log_prob, aux

def get_actor_continuous_action_from_posterior_vmap(
    posterior: BlockPosterior,
    actor: nnx.Module,
    critic1: nnx.Module,
    critic2: nnx.Module,
    obs: jax.Array,
    action_scale: jax.Array,
    action_bias: jax.Array,
    should_explore: bool,
    explore_n_samples: int,
    key: jax.Array,
) -> tuple[jax.Array, dict]:
    
    if should_explore:
        graphdef, _ = nnx.split(actor)
        
        def sample_and_apply(key):
            sub_key1, sub_key2 = jax.random.split(key)
            flat_state = block_sample(posterior, sub_key1)
            actor_state = _construct_state_from_flat_state(flat_state, posterior.shapes)
            temp_actor = nnx.merge(graphdef, actor_state)

            action, log_prob, mean = get_continuous_actor_action(
                temp_actor, obs, action_scale, action_bias, sub_key2
            )
            critics_input = jnp.concatenate([obs, action], axis=1)
            q_values = jnp.minimum(
                critic1(critics_input),
                critic2(critics_input),
            )
            return {'q_values': q_values.reshape(-1), 'action': action, 'log_prob': log_prob, 'aux': mean}
            
        keys = jax.random.split(key, explore_n_samples)
        results = jax.vmap(sample_and_apply)(keys)
        best_idx = jnp.argmax(results['q_values'], axis=0)
        
        action = results['action'][best_idx, jnp.arange(best_idx.shape[0])]
        log_prob = results['log_prob'][best_idx, jnp.arange(best_idx.shape[0])]
        aux = results['aux'][best_idx, jnp.arange(best_idx.shape[0])]
    else:
        action, log_prob, aux = get_continuous_actor_action(
            actor, obs, action_scale, action_bias, key
        )

    return action, log_prob, aux
