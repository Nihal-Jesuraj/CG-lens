"""Tune lambda (M1) and lambda, mu (M2) per (method, R, SNR) on recon_val ONLY, with the edge-extension rule; freeze to hyperparams.json."""
import argparse
import datetime
import json
import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from src import common, metrics

QUICK = dict(M1=[10.0 ** e for e in (-3, -2, -1, 0)], M2_lam=[1e-3, 1e-2, 1e-1], M2_mu=[1e-2, 1e-3])


def _psnr_one(x, idx, method, R, snr, params):
    xh, _ = common.reconstruct(x, idx, method, R, snr, params)
    return metrics.psnr(xh, x)


def mean_psnr(method, R, snr, params, idxs, img, n_jobs):
    vals = Parallel(n_jobs=n_jobs)(delayed(_psnr_one)(img[i], i, method, R, snr, params) for i in idxs)
    return float(np.mean(vals))


def _extend(grid, side):
    lo, hi = min(grid), max(grid)
    return [lo * 10 ** (-0.5 * k) for k in (1, 2)] if side == "low" else [hi * 10 ** (0.5 * k) for k in (1, 2)]


def tune_cell(method, R, snr, idxs, img, lams, mus, n_jobs):
    rows, evaluated = [], {}

    def run(lam_list, rnd):
        for lam in lam_list:
            for mu in mus:
                p = dict(lam=lam) if method == "M1" else dict(lam=lam, mu=mu)
                v = mean_psnr(method, R, snr, p, idxs, img, n_jobs)
                evaluated[(lam, mu)] = v
                rows.append(dict(method=method, R=R, snr=snr if math.isfinite(snr) else "inf", lam=lam, mu=mu if method == "M2" else np.nan,
                                 psnr_mean=v, n_slices=len(idxs), round=rnd))
    grid = list(lams)
    run(grid, 0)
    ext, edge_hit = 0, False
    while ext < config.EDGE_MAX_EXTENSIONS:
        best = max(evaluated, key=evaluated.get)
        lam_all = sorted({k[0] for k in evaluated})
        if best[0] == lam_all[0]:
            side = "low"
        elif best[0] == lam_all[-1]:
            side = "high"
        else:
            break
        edge_hit = True
        ext += 1
        new = _extend(lam_all, side)
        print(f"  edge rule: {method} R={R} SNR={snr}: best lam on {side} edge -> evaluating {['%.2e' % v for v in new]}", flush=True)
        run(new, ext)
    best = max(evaluated, key=evaluated.get)
    lam_all = sorted({k[0] for k in evaluated})
    at_edge_final = best[0] in (lam_all[0], lam_all[-1])
    out = dict(lam=best[0], psnr_mean=evaluated[best], extensions=ext, edge_hit=edge_hit, still_at_edge=at_edge_final)
    if method == "M2":
        out["mu"] = best[1]
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    config.ensure_dirs()
    hp = config.RESULTS_DIR / "hyperparams.json"
    if hp.exists() and not a.force:
        raise SystemExit("hyperparams.json is frozen. Use --force --reason '...' (logged); then re-run everything downstream.")
    if hp.exists() and a.force:
        if not a.reason:
            raise SystemExit("--force requires --reason")
        log_p = config.RESULTS_DIR / "freeze_log.json"
        log = json.loads(log_p.read_text()) if log_p.exists() else []
        log.append(dict(when=datetime.datetime.now(datetime.timezone.utc).isoformat(), reason=a.reason, previous_hash=json.dumps(json.loads(hp.read_text()), sort_keys=True)[:64]))
        log_p.write_text(json.dumps(log, indent=1))
    df, img, msk, splits = common.load_all()
    idxs = splits["recon_val"]["slice_index"]
    Rs, snrs = config.R_LIST, config.SNR_LIST
    m1, m2l, m2m = config.M1_LAMBDAS, config.M2_LAMBDAS, config.M2_MUS
    if a.quick:
        idxs = common.pick_quick(df, idxs, config.QUICK_SLICES // 2)
        Rs, m1, m2l, m2m = [config.R_PRIMARY], QUICK["M1"], QUICK["M2_lam"], QUICK["M2_mu"]
    hyper, table = dict(M1={}, M2={}), []
    for R in Rs:
        for snr in snrs:
            for method, lams, mus in (("M1", m1, [None]), ("M2", m2l, m2m)):
                print(f"tuning {method} R={R} SNR={snr} on {len(idxs)} recon_val slices", flush=True)
                best, rows = tune_cell(method, R, snr, idxs, img, lams, mus, a.n_jobs)
                hyper[method][common.cell_key(R, snr)] = best
                table += rows
    pd.DataFrame(table).to_csv(config.RESULTS_DIR / "tuning_table.csv", index=False)
    hyper.update(frozen_on="recon_val", n_slices=len(idxs), config_hash=config.config_hash(), quick=a.quick)
    common.write_json(hp, hyper)
    hp_rows = [dict(method=method, cell=cell, **values)
               for method in ("M1", "M2") for cell, values in hyper[method].items()]
    pd.DataFrame(hp_rows).to_csv(config.RESULTS_DIR / "hyperparameters.csv", index=False)
    print(json.dumps(hyper, indent=1))


if __name__ == "__main__":
    main()
