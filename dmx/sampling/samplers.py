import math

import jax
import numpy as np

from flax import nnx
from jax.typing import ArrayLike

jnp = jax.numpy

def edm_sigma_steps(sigma_min, sigma_max, num_steps, *, rho=7):
    sigma_steps = np.linspace(
        sigma_max ** (1 / rho), 
        sigma_min ** (1 / rho), 
        num_steps, 
    ) ** rho

    return sigma_steps


def vp_sigma_steps(sigma_min, sigma_max, num_steps, *, epsilon_s = 1e-3):
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    orig_t_steps = np.linspace(1, epsilon_s, num_steps)
    sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)

    return sigma_steps


def ve_sigma_steps(sigma_min, sigma_max, num_steps):
    step_indices = np.linspace(0, 1, num_steps)
    sigma_steps = (sigma_min / sigma_max) ** step_indices * sigma_max

    return sigma_steps


def iddpm_sigma_steps(sigma_min, sigma_max, num_steps, *, C_1=0.001, C_2=0.008, M=1000):
    step_indices = np.linspace(0, 1, num_steps)
    u = np.zeros(M + 1)
    alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
    for j in np.arange(M, 0, -1): # M, ..., 1
        u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
    u_filtered = u[(u >= sigma_min) & (u <= sigma_max)]
    sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).astype(int)]

    return sigma_steps


def edm_sampler(
    model:callable, 
    latents:ArrayLike,
    *,
    num_steps:int=18, 
    sigma_min:float=0.002, 
    sigma_max:float=80,
    solver:str='heun', 
    discretization:str='edm',
    S_churn:float=0, 
    S_min:float=0, 
    S_max:float=float('inf'), 
    S_noise:float=1,
    rngs:nnx.Rngs=nnx.Rngs(42),
    **kwargs,
):
    """ A geneator function implementing DM sampling
    
    A rewrite of the EDM sampler, with simplified parameterization and generalization.
    We no longer discretize t explicitly, since the function is invariant over sigma(t). 
    Instead we integrate over sigma and provide a few build-in options for 
    sigma discredization. Accordingly, S_min and S_max are defined on support of sigma instead 
    of t.

    Args:
        model: denoise model
        latents: inital noise input with unit sigma
    Keyward Args:
        num_steps: number of integration steps 
        sigma_min: minimal sigma discretization
        sigma_max=80: maxmimal sigma discretization
        solver: 'heun'|'euler'
        discretization: 'edm'|'vp'|'ve'|'iddpm'|custum discreization function
        S_churn: scale of noise churning, 0 will disable it 
        S_min: sigma range for churning, min value
        S_max: sigma range for churning, max value
        S_noise=1: noise scaling during churning
        rngs=nnx.Rngs(42): rng seed, only used during churn
        **kwargs: additional argument will be passed to model
    
    Yields:
        integration result at each discretization step        
    """
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']

    # Time step discretization.

    if discretization == "edm":
        discretization_fn = edm_sigma_steps
    elif discretization == "ve":
        discretization_fn = ve_sigma_steps
    elif discretization == "iddpm":
        discretization_fn = iddpm_sigma_steps
    elif discretization == 'vp':
        discretization_fn = vp_sigma_steps
    else:
        discretization_fn = discretization

    sigma_steps = discretization_fn(sigma_min, sigma_max, num_steps)
    sigma_steps = jnp.r_[sigma_steps, 0] # t_N = 0

    # Main sampling loop.
    gamma = min(S_churn / num_steps, math.sqrt(2) - 1)
    x_next = jnp.asarray(latents) * sigma_steps[0]
    for sigma_cur, sigma_next in zip(sigma_steps[:-1], sigma_steps[1:]):
        x_cur = x_next

        # Increase noise temporarily.
        if S_min <= sigma_cur <= S_max and gamma > 0:
            sigma_hat = sigma_cur * (1 +  gamma)
            e = jax.random.normal(rngs.sample(), x_cur.shape)
            x_hat = x_cur + S_noise * math.sqrt(sigma_hat ** 2 - sigma_cur ** 2) * e
        else:
            sigma_hat, x_hat = sigma_cur, x_cur

        # Euler step.
        d_cur = model(x_hat, sigma_hat, return_e=True, **kwargs)
        x_next = x_hat + (sigma_next - sigma_hat) * d_cur

        # Apply 2nd order correction.
        if solver == "heun" and sigma_next > 0:
            d_prime = model(x_next, sigma_next, return_e=True, **kwargs)
            x_next = x_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        yield x_next
