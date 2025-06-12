import jax

import pytest

from dmx.modeling.unet import DhariwalUNet
from dmx.modeling.precondition import EDMPrecond


def test_edm_precond():
    """Test the EDM Preconditioning module."""
    base_model = DhariwalUNet(out_channels=1)

    model = EDMPrecond(base_model=base_model, use_fp16=False)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 1))

    state = model.init(jax.random.PRNGKey(0), x, method="train")

    output = model.apply(state, x, method="train", rngs={"sigma": jax.random.PRNGKey(1)})

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"
