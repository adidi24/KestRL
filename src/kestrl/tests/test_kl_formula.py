import jax
import jax.numpy as jnp
import time

class LR_DiagonalGaussian:
    """ Low-rank diagonal Gaussian distribution. """
    def __init__(self, mean, std, P=None):
        self.mean = mean
        self.std = std
        self.P = P

    def sample(self, key):
        """ Sample from the distribution using JAX key splitting. """
        D = self.mean.shape[0]
        
        if self.P is None:
            eps = jax.random.normal(key, shape=(D,))
            return self.mean + self.std * eps
        else:
            # Split the key to ensure independent noise
            key_diag, key_lr = jax.random.split(key)
            
            K = self.P.shape[1]
            eps_diag = jax.random.normal(key_diag, shape=(D,))
            eps_lr = jax.random.normal(key_lr, shape=(K,))
            
            return self.mean + (self.std * eps_diag) + (self.P @ eps_lr)
    
    def kl(self, other):
        """ Efficient KL divergence KL(self || other) """
        if not isinstance(other, LR_DiagonalGaussian):
            raise ValueError("Other distribution must be of type LR_DiagonalGaussian")
            
        if self.P is not None and other.P is None:
            D = self.mean.shape[0]
            
            # --- Log-Determinant Term ---
            # Efficiently compute P^T * diag(std_q^-2) * P without DxD matrix
            inv_var_q = self.std ** -2
            M = jnp.eye(self.P.shape[1]) + self.P.T @ (inv_var_q[:, None] * self.P)
            _, logdet_M = jnp.linalg.slogdet(M)
            
            log_det_term = jnp.sum(jnp.log(other.std**2)) - jnp.sum(jnp.log(self.std**2)) - logdet_M
            
            # --- Constant Term ---
            constant_term = -D
            
            # --- Quadratic Term ---
            quad_term = jnp.sum(((self.mean - other.mean)**2) / (other.std**2))
            
            # --- Trace Term ---
            # Efficiently compute P^T * diag(std_p^-2) * P 
            inv_var_p = other.std ** -2
            trace_core = jnp.trace(self.P.T @ (inv_var_p[:, None] * self.P))
            trace_term = jnp.sum((self.std**2) / (other.std**2)) + trace_core
            
            kl = 0.5 * (log_det_term + constant_term + quad_term + trace_term)
            return kl
        else:
            raise NotImplementedError("Currently only supports Posterior with P and Prior without P.")
        
def validate_kl_divergence():
    key = jax.random.PRNGKey(42)
    key_mean_q, key_std_q, key_p, key_mean_p, key_std_p = jax.random.split(key, 5)
    
    # Dimensions
    D = 10
    K = 3
    
    # Initialize Prior (p)
    mean_p = jax.random.normal(key_mean_p, (D,))
    std_p = jnp.abs(jax.random.normal(key_std_p, (D,))) + 0.1 # Ensure positive
    prior = LR_DiagonalGaussian(mean=mean_p, std=std_p, P=None)
    
    # Initialize Posterior (q)
    mean_q = jax.random.normal(key_mean_q, (D,))
    std_q = jnp.abs(jax.random.normal(key_std_q, (D,))) + 0.1 # Ensure positive
    P_q = jax.random.normal(key_p, (D, K))
    posterior = LR_DiagonalGaussian(mean=mean_q, std=std_q, P=P_q)
    
    # ---------------------------------------------------------
    # 1. Calculate KL using our highly efficient method
    # ---------------------------------------------------------
    efficient_kl = posterior.kl(prior)
    
    # ---------------------------------------------------------
    # 2. Calculate KL using Brute Force Full Covariance Math
    # ---------------------------------------------------------
    # Build the full DxD covariance matrices
    Sigma_q = jnp.diag(posterior.std**2) + posterior.P @ posterior.P.T
    Sigma_p = jnp.diag(prior.std**2)
    
    # Brute force inverses and determinants
    inv_Sigma_p = jnp.diag(1.0 / prior.std**2) # Fast inverse for diagonal
    _, logdet_p = jnp.linalg.slogdet(Sigma_p)
    _, logdet_q = jnp.linalg.slogdet(Sigma_q)
    
    # Brute force components
    bf_log_det = logdet_p - logdet_q
    bf_quad = (posterior.mean - prior.mean).T @ inv_Sigma_p @ (posterior.mean - prior.mean)
    bf_trace = jnp.trace(inv_Sigma_p @ Sigma_q)
    
    brute_force_kl = 0.5 * (bf_log_det - D + bf_quad + bf_trace)
    
    # ---------------------------------------------------------
    # 3. Compare Results
    # ---------------------------------------------------------
    print(f"Efficient KL:    {efficient_kl:.6f}")
    print(f"Brute Force KL:  {brute_force_kl:.6f}")
    
    # JAX uses float32 by default, so we check for closeness with a small tolerance
    assert jnp.allclose(efficient_kl, brute_force_kl, atol=1e-5), "Mismatch between efficient and brute force KL!"
    print("✅ TEST PASSED: Efficient KL matches Full Covariance exactly.")

    # 4. Test Sampling
    sample_key = jax.random.PRNGKey(100)
    w = posterior.sample(sample_key)
    assert w.shape == (D,), f"Sample shape incorrect. Expected ({D},), got {w.shape}"
    print("✅ TEST PASSED: Sampling works and maintains correct dimensions.")


def test_kl_p_zero_matches_diagonal():
    """When P=0, block KL must equal the closed-form diagonal KL exactly."""
    key = jax.random.PRNGKey(7)
    km, ks, kmp, ksp = jax.random.split(key, 4)

    D = 32
    mean_q = jax.random.normal(km,  (D,))
    std_q  = jnp.abs(jax.random.normal(ks,  (D,))) + 0.1
    mean_p = jax.random.normal(kmp, (D,))
    std_p  = jnp.abs(jax.random.normal(ksp, (D,))) + 0.1

    # Block posterior with P identically zero
    P_zero = jnp.zeros((D, 5))
    posterior = LR_DiagonalGaussian(mean=mean_q, std=std_q, P=P_zero)
    prior     = LR_DiagonalGaussian(mean=mean_p, std=std_p, P=None)
    block_kl  = posterior.kl(prior)

    # Closed-form diagonal KL: 0.5 * sum(log(σ_p²/σ_q²) + σ_q²/σ_p² + (μ_q-μ_p)²/σ_p² - 1)
    diag_kl = 0.5 * jnp.sum(
        2 * (jnp.log(std_p) - jnp.log(std_q))
        + (std_q**2 + (mean_q - mean_p)**2) / std_p**2
        - 1.0
    )

    assert jnp.allclose(block_kl, diag_kl, atol=1e-5), (
        f"P=0 case mismatch: block={block_kl:.6f}, diagonal={diag_kl:.6f}"
    )
    print("✅ TEST PASSED: P=0 block KL matches closed-form diagonal KL exactly.")


def test_kl_is_nonneg():
    """KL(posterior || prior) must be >= 0 for all random configurations."""
    D, K = 20, 4
    failures = []
    for seed in range(50):
        key = jax.random.PRNGKey(seed)
        km, ks, kp, kmp, ksp = jax.random.split(key, 5)
        posterior = LR_DiagonalGaussian(
            mean=jax.random.normal(km,  (D,)),
            std =jnp.abs(jax.random.normal(ks,  (D,))) + 0.05,
            P   =jax.random.normal(kp,  (D, K)) * 0.1,
        )
        prior = LR_DiagonalGaussian(
            mean=jax.random.normal(kmp, (D,)),
            std =jnp.abs(jax.random.normal(ksp, (D,))) + 0.05,
            P   =None,
        )
        kl_val = float(posterior.kl(prior))
        if kl_val < -1e-5:   # allow tiny float32 rounding
            failures.append((seed, kl_val))

    assert not failures, f"KL was negative for seeds: {failures}"
    print(f"✅ TEST PASSED: KL >= 0 for all 50 random seeds.")


def kl_pure_math(mean_q, std_q, P_q, mean_p, std_p):
    D = mean_q.shape[0]
    
    # Log-Determinant
    inv_var_q = std_q ** -2
    M = jnp.eye(P_q.shape[1]) + P_q.T @ (inv_var_q[:, None] * P_q)
    _, logdet_M = jnp.linalg.slogdet(M)
    log_det_term = jnp.sum(jnp.log(std_p**2)) - jnp.sum(jnp.log(std_q**2)) - logdet_M
    
    # Constant & Quadratic
    constant_term = -D
    quad_term = jnp.sum(((mean_q - mean_p)**2) / (std_p**2))
    
    # Trace
    inv_var_p = std_p ** -2
    trace_core = jnp.trace(P_q.T @ (inv_var_p[:, None] * P_q))
    trace_term = jnp.sum((std_q**2) / (std_p**2)) + trace_core
    
    return 0.5 * (log_det_term + constant_term + quad_term + trace_term)

# 2. Create the JIT compiled version
jit_kl_pure_math = jax.jit(kl_pure_math)

def run_benchmark():
    # Setup Data (Simulating a decently sized layer: 10,000 parameters, 10 global factors)
    # D, K = 10000, 10
    D, K = 65536, 20
    key = jax.random.PRNGKey(0)
    mean_q, std_q, P_q, mean_p, std_p = jax.random.split(key, 5)
    
    mean_q = jax.random.normal(mean_q, (D,))
    std_q = jnp.abs(jax.random.normal(std_q, (D,))) + 0.1
    P_q = jax.random.normal(P_q, (D, K))
    mean_p = jax.random.normal(mean_p, (D,))
    std_p = jnp.abs(jax.random.normal(std_p, (D,))) + 0.1

    print(f"Benchmarking KL Divergence (D={D}, K={K})...")

    # --- WARMUP ---
    # We run JIT once without timing it, because the first run includes compilation time
    _ = jit_kl_pure_math(mean_q, std_q, P_q, mean_p, std_p).block_until_ready()

    n_runs = 1000

    # --- UNJITTED TIMING ---
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = kl_pure_math(mean_q, std_q, P_q, mean_p, std_p).block_until_ready()
    unjitted_time = (time.perf_counter() - start) / n_runs

    # --- JITTED TIMING ---
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = jit_kl_pure_math(mean_q, std_q, P_q, mean_p, std_p).block_until_ready()
    jitted_time = (time.perf_counter() - start) / n_runs

    print("-" * 40)
    print(f"Unjitted Time per run: {unjitted_time * 1000:.4f} ms")
    print(f"Jitted Time per run:   {jitted_time * 1000:.4f} ms")
    print(f"Speedup Factor:        {unjitted_time / jitted_time:.1f}x 🚀")
    

if __name__ == "__main__":
    validate_kl_divergence()
    test_kl_p_zero_matches_diagonal()
    test_kl_is_nonneg()
    print()
    run_benchmark()