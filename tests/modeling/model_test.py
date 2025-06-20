import jax

import pytest

from flax import nnx
from dmx.modeling.unet import DhariwalUNet
from dmx.modeling.uvit import UVit

jnp = jax.numpy


def test_uvit():
    """Test the uvit UNet model."""
    rngs = nnx.Rngs(1)
    model = UVit(input_shape=(256,256,1), out_channels=1, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 256, 256, 1))
    noise = jax.random.uniform(jax.random.PRNGKey(1), (2,))
    
    output = model(x, noise, class_labels=None, deterministic=True)

    assert output.shape == (2, 256, 256, 1), "Output shape mismatch"


def test_dhariwa_unet():
    """Test the Dhariwal UNet model."""
    rngs = nnx.Rngs(1)

    model = DhariwalUNet(in_channels=1, out_channels=1, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 1))
    noise = jax.random.uniform(jax.random.PRNGKey(1), (2,))
    
    output = model(x, noise, class_labels=None, deterministic=True)

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"
