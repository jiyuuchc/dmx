import numpy as np

def clean_up_mask(mask):
    """ ensure continuity of mask ID"""
    unique_ids = np.unique(mask)
    lut = np.zeros(unique_ids.max()+1, dtype=int)
    lut[unique_ids] = np.arange(len(unique_ids))

    return lut[mask]
