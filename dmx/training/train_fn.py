import flax.linen as nn
import jax
import jax.numpy as jnp

from dmx.modeling.precondition import edm_precond

def emd_train_fn(model, x, *, sigma_min=0.002, sigma_max=80, sigma_data=0.5, p_mean=-1.2, p_std=1.2, **kwargs):
    """Compute the training loss for the EMD model."""
    B, H, W, C = x.shape

    log_sigma = jax.random.normal(
        model.make_rng("sigma"), 
        (B, 1, 1, 1), 
    ) * p_std + p_mean

    sigma = jnp.clip(
        jnp.exp(log_sigma),
        sigma_min,
        sigma_max,
    )

    y = x + jax.random.normal(
        model.make_rng("sigma"), 
        (B, H, W, C),
    ) * sigma

    D_x = edm_precond(model, y, sigma=sigma, sigma_data=sigma_data, deterministic=False, **kwargs)

    # Compute the loss
    loss_weight = 1 / (sigma ** 2) + 1 / (sigma_data ** 2) 
    loss = ((x - D_x) ** 2) * loss_weight

    return loss
