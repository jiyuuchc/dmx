import jax
import numpy as np
import pytest

jpn = jax.numpy

def test_flow():
    from dmx.utils.flow import mask_to_flow

    mask = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])

    flows = mask_to_flow(mask)

    expected_flows = np.array([
        [[ 0.,  0.],
        [ 1.,  0.],
        [ 0.,  0.]],

       [[ 0.,  1.],
        [ 0.,  0.],
        [ 0., -1.]],

       [[ 0.,  0.],
        [-1.,  0.],
        [ 0.,  0.]]])
    
    assert np.allclose(flows, expected_flows), "Flow computation mismatch"


def test_follow_flows():
    from dmx.utils.flow import follow_flows

    dP = np.array([[[0.1, 0.2],
                    [0.3, 0.4]],

                   [[-0.1, -0.2],
                    [-0.3, -0.4]]])

    p = follow_flows(dP, niter=50)

    expected = np.array(
        [[[ 0.5       ,  0.81870174],
        [ 0.4999983 ,  1.6666644 ]],

       [[ 0.54343927, -0.91312176],
        [ 0.50000006,  0.29791135]]])
    
    assert np.allclose(p, expected), "Follow flows computation mismatch"


def test_mask_recovery():
    from dmx.utils.flow import mask_to_flow, flow_to_mask

    mask = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])

    flows = mask_to_flow(mask)

    recovered_mask = flow_to_mask(flows, min_seed_cnts=1)

    assert np.array_equal(recovered_mask, mask), "Mask recovery from flow failed"
