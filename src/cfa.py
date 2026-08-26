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

XTRANS_PATTERN = np.ascontiguousarray(np.roll(XTRANS_PATTERN, (1, 2), axis=(0, 1)))

BAYER_PATTERN = np.array([
    [0, 1],
    [1, 2]
], dtype=np.int32)


def get_mosaic_arr(shape: tuple, pattern) -> tuple:
    """
        Return 3 arrays which are used to project an image (H, W, 3) onto mosaic pattern (H, W). Example:
            yy, xx, cfa_indices = get_mosaic_arr(img.shape, XTRANS_PATTERN)
            mosaiced = img[yy, xx, cfa_indices]

        Returns:
            tuple(yy, xx, cfa_indices)
    """
    H, W = shape[0], shape[1]
    ph, pw = pattern.shape
    yy, xx = np.ogrid[:H, :W]
    yy = yy.astype(np.int32)
    xx = xx.astype(np.int32)
    yy_mod = yy % ph
    xx_mod = xx % pw
    cfa_indices = pattern[yy_mod, xx_mod]

    return yy, xx, cfa_indices
