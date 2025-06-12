import flax.linen as nn
import jax
import jax.numpy as jnp


def edm_precond(module, x, sigma, sigma_data = 0.5, *, return_dx=True, **kwargs):
    """ Preconditioning function accoding to the EDM paper.
    Args:
        module: base model e.g., DharwaUNet
        x: image data with noise (B, H, W, C)
        sigma: noise amplitude (B,)
        sigma_data: signal amplitude
    KeywardArgs:
        return_dx: if True, return D(x) else e(x) = (x - D(x)) / sigma
    
    Returns:
        D(x) if return_dx is True, else e(x) (B, H, W, C)
    """
    dtype = module.dtype or jnp.float32

    sigma = sigma.reshape(-1, 1, 1, 1) 
    sigma_data = jnp.array(sigma_data, dtype=sigma.dtype).reshape(-1, 1, 1, 1)

    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = jnp.sqrt(sigma * sigma_data / (sigma ** 2 + sigma_data ** 2))
    c_in = 1 / jnp.sqrt(sigma ** 2 + sigma_data ** 2)

    c_noise = jnp.log(sigma.reshape(-1) / (2 * sigma_data)) / 4 

    F_x = module(
        (c_in * x).astype(dtype), 
        c_noise.astype(dtype),
        **kwargs,
    )

    D_x = c_skip * x + c_out * F_x.astype("float32")

    if return_dx:
        return D_x
    else:
        return (x - D_x) / sigma
