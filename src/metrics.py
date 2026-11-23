"""PSNR (MAX=1), SSIM (data_range=1), NMSE, ROI-NMSE (Convention 12)."""
import numpy as np
from skimage.metrics import structural_similarity

import config


def nmse(x_hat, x):
    return float(np.sum((x_hat - x) ** 2) / np.sum(x ** 2))


def psnr(x_hat, x):
    mse = float(np.mean((x_hat - x) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def ssim(x_hat, x):
    return float(structural_similarity(x, x_hat, data_range=1.0))


def roi_eligible(mask):
    return int(np.count_nonzero(mask)) >= config.MIN_ROI_PIXELS


def roi_nmse(x_hat, x, mask):
    """NMSE over tumorMask pixels (128x128 mask); enforces MIN_ROI_PIXELS."""
    sel = np.asarray(mask) > 0
    if int(sel.sum()) < config.MIN_ROI_PIXELS:
        raise ValueError(f"ROI has {int(sel.sum())} < MIN_ROI_PIXELS={config.MIN_ROI_PIXELS} pixels")
    return float(np.sum((x_hat[sel] - x[sel]) ** 2) / np.sum(x[sel] ** 2))


def all_metrics(x_hat, x, mask):
    return dict(psnr=psnr(x_hat, x), ssim=ssim(x_hat, x), nmse=nmse(x_hat, x), roi_nmse=roi_nmse(x_hat, x, mask))
