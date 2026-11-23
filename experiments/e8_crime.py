"""E8 (P1): inverse-crime sensitivity. k-space from a 256x256 area-averaged slice (central 128x128 block), reconstruct on the 128 grid."""
import argparse

import numpy as np
import pandas as pd

import config
from src import common, data, metrics
from src import mri_ops as mo
from src import recon as rc


def hires_spectrum(x256):
    """Central 128x128 block of the 256x256 ortho spectrum, FFT order, conjugate-symmetrised, rescaled (x 1/2) to match mean intensity."""
    S = np.fft.fftshift(mo.fft2c(x256))
    c = np.fft.ifftshift(S[64:192, 64:192])
    c = 0.5 * (c + np.conj(mo.partner(c)))        # restores exact symmetry on the Nyquist row/column
    return 0.5 * c                                 # ortho scaling: 256 -> 128 changes the mean by a factor 2


def observe_hires(x256, R, snr, idx):
    N = config.N
    crop = hires_spectrum(x256)
    M = mo.make_mask(N, R, (R, idx))
    Ax = M * crop
    return M, Ax + mo.make_noise(Ax, M, snr, (R, config.snr_code(snr), idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()
    df, img, msk, splits = common.load_all()
    hyper = common.load_hyper()
    idxs = splits["recon_eval"]["slice_index"]
    snrs = config.SNR_LIST
    if a.quick:
        idxs, snrs = common.pick_quick(df, idxs, 8), [config.SNR_PRIMARY]
    hi = data.extract_hires(df, idxs, 256)
    R = config.R_PRIMARY
    rows = []
    for i in idxs:
        x = img[i]
        native = int(df.loc[df["index"] == i, "h"].iloc[0])
        for snr in snrs:
            M, y = observe_hires(hi[i], R, snr, i)
            for m in common.METHODS:
                p = common.params_for(hyper, m, R, snr)
                xm, _ = common.reconstruct(x, i, m, R, snr, p)
                xc = rc.zero_filled(y, M) if m == "M0" else (rc.tikhonov_cg(y, M, p["lam"])[0] if m == "M1" else rc.tv_irls_cg(y, M, p["lam"], p["mu"])[0])
                rows.append(dict(slice_index=i, native_size=native, method=m, snr=snr if np.isfinite(snr) else "inf", psnr_matched=metrics.psnr(xm, x), ssim_matched=metrics.ssim(xm, x),
                                 psnr_crime_free=metrics.psnr(xc, x), ssim_crime_free=metrics.ssim(xc, x)))
        print("slice", i, "done", flush=True)
    r = pd.DataFrame(rows)
    r["d_psnr"], r["d_ssim"] = r.psnr_crime_free - r.psnr_matched, r.ssim_crime_free - r.ssim_matched
    r.to_csv(config.RESULTS_DIR / "e8_crime.csv", index=False)
    s = r.groupby(["method", "snr"], as_index=False).mean(numeric_only=True)
    s["rank_matched"] = s.groupby("snr").psnr_matched.rank(ascending=False)
    s["rank_crime_free"] = s.groupby("snr").psnr_crime_free.rank(ascending=False)
    s.drop(columns=["slice_index"]).to_csv(config.RESULTS_DIR / "e8_summary.csv", index=False)
    print(s.drop(columns=["slice_index"]).to_string(index=False))


if __name__ == "__main__":
    main()
