import jax

import pytest

from dmx.modeling.unet import DhariwalUNet
from dmx.modeling.precondition import edm_precond


def test_edm_precond():
    """Test the EDM Preconditioning module."""
    model = DhariwalUNet(out_channels=1)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 1))
    sigma = jax.random.uniform(jax.random.PRNGKey(1), (2,))

    state = model.init(jax.random.PRNGKey(0), x, sigma, method=edm_precond)

    output = model.apply(state, x, sigma, method=edm_precond)

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"
