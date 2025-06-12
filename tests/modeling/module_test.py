import jax
import numpy as np
import pytest

from dmx.modeling.common import LabelEmbedding

def test_label_embedding():
    x = np.array([0.1, 0.2, 0.3])
    m = LabelEmbedding(1024)
    state = m.init(jax.random.PRNGKey(1), x)

    emb = m.apply(state, x)

    assert emb.shape == (3, 1024)

    emb = m.apply(state, x[:, None, None, None])

    assert emb.shape == (3, 1024)
