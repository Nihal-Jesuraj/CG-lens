import numpy as np
import pytest

from src import metrics as me


def test_basic_values():
    x = np.random.default_rng(0).random((32, 32))
    assert me.psnr(x, x) == float("inf") and me.nmse(x, x) == 0 and abs(me.ssim(x, x) - 1) < 1e-12
    xh = x + 0.1
    assert abs(me.psnr(xh, x) - 20.0) < 1e-9          # MSE = 0.01, MAX = 1
    assert abs(me.nmse(xh, x) - 0.01 * x.size / np.sum(x ** 2)) < 1e-12


def test_roi_nmse_uses_mask_and_min_pixels():
    x = np.ones((32, 32))
    xh = x.copy()
    mask = np.zeros((32, 32))
    mask[:8, :8] = 1
    xh[:8, :8] = 1.5
    assert abs(me.roi_nmse(xh, x, mask) - 0.25) < 1e-12
    small = np.zeros((32, 32))
    small[:3, :3] = 1
    with pytest.raises(ValueError):
        me.roi_nmse(xh, x, small)
    assert me.roi_eligible(mask) and not me.roi_eligible(small)
