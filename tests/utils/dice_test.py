import numpy as np
from dmx.utils.dice import Dice

def test_empty_mask():
    m = Dice()
    mask = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
    empty_mask = np.zeros_like(mask)

    m.update(mask, empty_mask)

    m.update(empty_mask, mask)

    m.update(empty_mask, empty_mask)

    assert m.compute() == 0
