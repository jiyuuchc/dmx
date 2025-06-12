import flax.linen as nn
import jax
import jax.numpy as jnp


class EDMPrecond(nn.Module):
    """ Preconditioning module for the EDM paper.
        https://github.com/openai/guided-diffusion

    Parameters:
        base_model: base model e.g. DhariwalUNet.
        use_fp16: whether to use float16 precision.
        sigma_min: minimum sigma value
        sigma_max: maximum sigma value
        sigma_data: sigma value for data noise
    """
    base_model: nn.Module
    use_fp16: bool = False
    sigma_min: float = 0.002
    sigma_max: float = 80
    sigma_data: float = 0.5

    def setup(self):
        if self.use_fp16:
            self.base_model = self.base_model.copy(dtype=jnp.float16)

    def train(self, y, class_labels=None):
        B, H, W, C = y.shape

        sigma = jax.random.uniform(
            self.make_rng("sigma"), 
            (B, 1, 1, 1), 
            minval=self.sigma_min, 
            maxval=self.sigma_max
        )
        n = jax.random.normal(
            self.make_rng("sigma"), 
            (B, H, W, C), 
        )

        x = y + n * sigma

        return self.__call__(x, sigma, class_labels=class_labels, deterministic=False)


    def __call__(self, x, sigma, class_labels=None, *, deterministic=True):
        sigma = sigma.reshape(-1, 1, 1, 1)

        model = self.base_model

        dtype = model.dtype or jnp.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = jnp.sqrt(sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2))
        c_in = jnp.sqrt(1 / (self.sigma_data ** 2 + sigma ** 2))
        c_noise = (jnp.log(sigma) / 4).reshape(-1)

        F_x = model(
            (c_in * x).astype(dtype), 
            c_noise.astype(dtype), 
            class_labels=class_labels,
            deterministic=deterministic,
        )

        assert F_x.dtype == dtype

        D_x = c_skip * x + c_out * F_x.astype("float32")

        return D_x

