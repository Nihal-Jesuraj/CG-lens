"""One command reproduces everything.   python run_all.py [--quick] [--synthetic] [--p1] [--n-jobs K]

--synthetic : smoke-test on a generated Cheng-like dataset (data/synthetic, results_synthetic). NOT scientific results.
--quick     : 16 slices, R = 4 only, reduced grids and folds (target: under 10 minutes on one core).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--p1", action="store_true", help="also run the P1 items (E2 c-e, E7, E8)")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--refreeze", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--with-tests", action="store_true")
    ap.add_argument("--audit-only", action="store_true", help="stop after the dataset audit so the counts can be reviewed")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.synthetic:
        env.update(RECONLENS_DATA=str(ROOT / "data" / "synthetic"), RECONLENS_RAW=str(ROOT / "data" / "synthetic" / "raw"),
                   RECONLENS_RESULTS=str(ROOT / "results_synthetic"), RECONLENS_SYNTHETIC="1")
        for d in (ROOT / "data" / "synthetic", ROOT / "results_synthetic"):
            shutil.rmtree(d, ignore_errors=True)
    raw = Path(env.get("RECONLENS_RAW", ROOT / "data" / "raw"))
    results = Path(env.get("RECONLENS_RESULTS", ROOT / "results"))
    log, t_all = [], time.time()

    def step(name, *cmd):
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, *cmd], env=env, cwd=ROOT)
        log.append(dict(step=name, seconds=round(time.time() - t0, 1), returncode=r.returncode))
        if r.returncode != 0:
            results.mkdir(parents=True, exist_ok=True)
            (results / "run_log.json").write_text(json.dumps(log, indent=1))
            raise SystemExit(f"step failed: {name}")

    q = ["--quick"] if a.quick else []
    nj = ["--n-jobs", str(a.n_jobs)]
    if a.with_tests:
        step("pytest", "-m", "pytest", "-q")
    if a.synthetic:
        step("synthetic data", "scripts/make_synthetic_data.py", str(raw))
    elif not list(raw.glob("*.mat")):
        raise SystemExit(f"No .mat files in {raw}. Download the Cheng dataset (figshare DOI 10.6084/m9.figshare.1512427) into data/raw/ first.")
    step("environment", "scripts/write_env.py")
    step("loader facts", "-m", "src.data", "verify")
    step("dataset audit", "-m", "src.data", "audit")
    step("dedup provenance", "scripts/audit_dedup.py", "--sample-count", "12", "--seed", "2026", "--n-permutations", "10000")
    if a.audit_only:
        print("audit written; review attrition.csv / roi_filter_audit.csv, then re-run without --audit-only")
        return
    step("cohorts", "-m", "src.data", "cohorts")
    step("data manifest", "-m", "src.data", "manifest")
    step("G-BENCH", "scripts/benchmark.py", "--n", "3" if a.quick else "10")
    hp = results / "hyperparams.json"
    if hp.exists() and not a.refreeze:
        print("hyperparams.json already frozen: tuning skipped")
    else:
        step("tuning (recon_val only)", "-m", "experiments.tune", *q, *nj, *(["--force", "--reason", a.reason or "refreeze"] if hp.exists() else []))
    step("E1 reconstruction grid", "-m", "experiments.e1_recon", *q, *nj)
    step("E1 serial timing pass", "-m", "experiments.e1_recon", "--timing", *q)
    step("E3 CVXOPT validation", "-m", "experiments.e3_cvx", *q)
    step("E2 CG convergence", "-m", "experiments.e2_cg", *q, *(["--p1"] if a.p1 else []))
    step("E4 diagnosis", "-m", "experiments.e4_diag", *q, *nj)
    step("E5 SVM solvers", "-m", "experiments.e5_svm", *q)
    if a.p1:
        step("E7 discrepancy lambda", "-m", "experiments.e7_lambda", *q)
        step("E8 inverse crime", "-m", "experiments.e8_crime", *q)
    step("validation summary", "scripts/validation_summary.py")
    step("figures", "scripts/make_figures.py")
    step("tables", "scripts/make_tables.py")
    step("manifest", "scripts/write_env.py")
    (results / "run_log.json").write_text(json.dumps(log, indent=1))
    # run_log.json is itself in the manifest, so refresh hashes after writing it.
    from scripts.write_env import write_manifest
    write_manifest()
    print(f"\nALL STEPS DONE in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
