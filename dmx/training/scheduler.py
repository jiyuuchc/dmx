import jax
import numpy as np
jnp = jax.numpy

def edm_schedule(key, x, *, sigma_data=0.5, sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2):
    B, H, W, C = x.shape
    log_sigma = jax.random.normal(
        key,
        (B, 1, 1, 1), 
    ) * p_std + p_mean

    sigma = jnp.clip(
        jnp.exp(log_sigma),
        sigma_min,
        sigma_max,
    )

    loss_weight = 1 / (sigma ** 2) + 1 / (sigma_data ** 2) 

    return sigma, loss_weight


def elbo_schedule(key, x, *, sigma_min=0.002, sigma_max=80):
    B, H, W, C = x.shape

    sigma = jax.random.uniform(
        key,
        (B, 1, 1, 1),
        minval=sigma_min,
        maxval=sigma_max,
    )

    loss_weight = 1

    return sigma, loss_weight


def train_with_schedule(key, model, x, schedule_fn=edm_schedule, **kwargs):
    key1, key2 = jax.random.split(key, 2)

    sigma, loss_weight = schedule_fn(key1, x)

    y = x + jax.random.normal(key2, x.shape) * sigma

    D_x = model(y, sigma=sigma, deterministic=False, **kwargs)

    loss = ((x - D_x) ** 2) * loss_weight

    return loss
