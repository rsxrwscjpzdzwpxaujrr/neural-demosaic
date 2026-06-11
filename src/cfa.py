import numpy as np

# Canonical patterns (R=0, G=1, B=2)
XTRANS_PATTERN = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], dtype=np.int32)

BAYER_PATTERN = np.array([
    [0, 1],
    [1, 2]
], dtype=np.int32)


def make_mosaic(rgb: np.ndarray, pattern: np.ndarray):
    """
    Projects an RGB image (H, W, 3) onto a mosaiced sensor image (H, W) using a CFA pattern.
    Pattern is a (ph, pw) ndarray with 0=R, 1=G, 2=B encoding the CFA. 
    Returns a (H, W) mosaiced array.
    """
    H, W = rgb.shape[0], rgb.shape[1]
    ph, pw = pattern.shape
    yy, xx = np.ogrid[:H, :W]
    yy_mod = yy % ph
    xx_mod = xx % pw
    cfa_indices = pattern[yy_mod, xx_mod]
    mosaic = rgb[yy, xx, cfa_indices]
    return mosaic