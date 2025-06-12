import jax

import pytest

from dmx.modeling.uvit import UVit

def test_uvit():
    """Test the Dhariwal UNet model."""
    model = UVit(out_channels=1)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 256, 256, 1))
    noise = jax.random.uniform(jax.random.PRNGKey(1), (2,))
    
    state = model.init(jax.random.PRNGKey(0), x, noise, class_labels=None, deterministic=False)

    output = model.apply(state, x, noise, class_labels=None, deterministic=True)

    assert output.shape == (2, 256, 256, 1), "Output shape mismatch"
