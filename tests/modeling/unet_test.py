import jax

import pytest

from dmx.modeling.unet import DhariwalUNet


def test_dhariwa_unet():
    """Test the Dhariwal UNet model."""
    model = DhariwalUNet(out_channels=1)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 1))
    noise = jax.random.uniform(jax.random.PRNGKey(1), (2,))
    
    state = model.init(jax.random.PRNGKey(0), x, noise, class_labels=None, deterministic=False)

    output = model.apply(state, x, noise, class_labels=None, deterministic=True)

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"
