import jax

import pytest

from flax import nnx

from dmx.modeling.unet import DhariwalUNet
from dmx.modeling.uvit import UVit
from dmx.modeling.precondition import EdmPrecond

jnp = jax.numpy

def test_edm():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 1))
    sigma = jax.random.uniform(jax.random.PRNGKey(1), (2, 1, 1, 1))

    base_model = DhariwalUNet(in_channels=1, out_channels=1, rngs=nnx.Rngs(1))
    model = EdmPrecond(base_model, rngs=nnx.Rngs(train=2))

    output = model(x, sigma)

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"

    output = model.train(x)
    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"

    base_model = UVit(input_shape=(64, 64, 1), out_channels=1, rngs=nnx.Rngs(1))
    model = EdmPrecond(base_model, rngs=nnx.Rngs(train=2))

    output = model(x, sigma)

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"

    output = model.train(x)
    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"
