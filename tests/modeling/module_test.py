import jax
import numpy as np
import pytest

from flax import nnx

from dmx.modeling.common import LabelEmbedding

def test_label_embedding():
    rngs = nnx.Rngs(1)

    x = np.array([0.1, 0.2, 0.3])
    m = LabelEmbedding(1024, label_dim=10,rngs=rngs)

    emb = m(x)

    assert emb.shape == (3, 1024)

    emb = m(x[:, None, None, None])

    assert emb.shape == (3, 1024)
