from dataclasses import dataclass, InitVar

from flax import nnx
import jax
jnp = jax.numpy


@dataclass(repr=False)
class EdmPrecond(nnx.Module):
    """ Preconditioning accoding to the EDM paper.
    Args:
        module: base model e.g., DharwaUNet
        sigma_data: signal amplitude
    KeywardArgs:
        return_dx: if True, return D(x) else e(x) = (x - D(x)) / sigma
    
    Returns:
        D(x) if return_dx is True, else e(x) (B, H, W, C)
    """
    module: nnx.Module
    sigma_data: float = 0.5
    sigma_min:float = 0.002
    sigma_max:float = 80
    p_mean:float = -1.2
    p_std:float = 1.2
    rngs: nnx.Rngs = nnx.Rngs(train=42)

    def __call__(self, x, sigma, x_cond=None, *, return_e:bool=False, **kwargs):
        """ Preconditioning function
        Args:
            x: image data with noise (B, H, W, C)
            sigma: noise amplitude (B, 1, 1, 1) or (B,) or ()
            x_cond: optional conditioning data (B, H, W, D)
        Keyward Args:
            return_e: if True, return D(x) else e(x) = (x - D(x)) / sigma
        
        Returns:
            D(x) if return_dx is True, else e(x) (B, H, W, C)
        """
        dtype = self.module.dtype or jnp.float32

        sigma_data = self.sigma_data
        sigma = jnp.asarray(sigma)

        if sigma.size == 1:
            sigma = jnp.repeat(sigma, x.shape[0])
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1, 1, 1)

        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = jnp.sqrt(sigma * sigma_data / (sigma ** 2 + sigma_data ** 2))
        c_in = 1 / jnp.sqrt(sigma ** 2 + sigma_data ** 2)

        c_noise = jnp.log(sigma.reshape(-1)) / 4 

        x_in = c_in * x
        if x_cond is not None:
            x_in = jnp.concatenate([x_in, x_cond], axis=-1)

        F_x = self.module(
            x_in.astype(dtype), 
            c_noise.astype(dtype),
            **kwargs,
        )

        D_x = c_skip * x + c_out * F_x.astype("float32")

        if not return_e:
            return D_x
        else:
            return (x - D_x) / sigma


    def train(self, x, x_cond=None, **kwargs):
        """Compute the training loss for the EMD paper
        Args:
            x: image data (B, H, W, C)
            x_cond: optional conditioning data (B, H, W, D)
        returns:
            loss: MSE (B, H, W, C)
        """
        B, H, W, C = x.shape

        log_sigma = jax.random.normal(
            self.rngs.train(),
            (B, 1, 1, 1), 
        ) * self.p_std + self.p_mean

        sigma = jnp.clip(
            jnp.exp(log_sigma),
            self.sigma_min,
            self.sigma_max,
        )

        y = x + jax.random.normal(
            self.rngs.train(), 
            (B, H, W, C),
        ) * sigma

        D_x = self(y, sigma=sigma, x_cond=x_cond, deterministic=False, **kwargs)

        # Compute the loss
        loss_weight = 1 / (sigma ** 2) + 1 / (self.sigma_data ** 2) 
        loss = ((x - D_x) ** 2) * loss_weight

        return loss
