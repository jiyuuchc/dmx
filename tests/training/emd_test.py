import jax

import pytest

from dmx.modeling.unet import DhariwalUNet
from dmx.training.train_fn import emd_train_fn

def test_edm():
    model = DhariwalUNet(out_channels=1)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 64, 1))

    state = model.init(jax.random.PRNGKey(0), x, method=emd_train_fn)

    rngs = dict(sigma=jax.random.PRNGKey(1), dropout=jax.random.PRNGKey(2))
    output = model.apply(state, x, method=emd_train_fn, rngs=rngs)

    assert output.shape == (2, 64, 64, 1), "Output shape mismatch"
