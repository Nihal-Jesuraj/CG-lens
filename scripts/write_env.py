"""Write results/env.json (CPU, cores, versions) and results/manifest.json (hashes of every output file)."""
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import cache


def cpu_model():
    override = os.environ.get("RECONLENS_CPU_MODEL")
    if override:
        return override
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                                    capture_output=True, text=True)
            if result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def write_env():
    config.ensure_dirs()
    pipeline = ["T1-CE slice +\ntumor mask", "undersampled\nnoisy k-space", "M0 / M1 / M2\nreconstruction",
                "fidelity: PSNR, SSIM,\nNMSE, ROI-NMSE", "64x64 features\n+ PCA (train only)",
                "SVM QP / SMO\npatient-disjoint CV", "fidelity vs\ndiagnosis"]
    pd.DataFrame({"order": range(1, len(pipeline) + 1), "stage": pipeline}).to_csv(config.RESULTS_DIR / "pipeline.csv", index=False)
    pd.DataFrame({"limitation": config.LIMITATIONS}).to_csv(config.RESULTS_DIR / "limitations.csv", index=False)
    libs = {}
    for m in ["numpy", "scipy", "h5py", "matplotlib", "pandas", "skimage", "sklearn", "cvxopt", "PIL", "joblib", "pytest"]:
        try:
            libs[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except ImportError:
            libs[m] = "not installed"
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip() or None
    except OSError:
        git = None
    env = dict(cpu=cpu_model(), cores=os.cpu_count(), python=sys.version, platform=platform.platform(), libraries=libs, git_commit=git,
               thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    (config.RESULTS_DIR / "env.json").write_text(json.dumps(env, indent=1))
    return env


def write_manifest():
    files = {}
    for p in sorted(config.RESULTS_DIR.rglob("*")):
        if p.is_file() and p.name not in {"manifest.json", ".DS_Store"} and "__pycache__" not in p.parts:
            files[str(p.relative_to(config.RESULTS_DIR))] = cache.file_sha256(p)
    man = dict(config_hash=config.config_hash(), global_seed=config.GLOBAL_SEED, synthetic_data=os.environ.get("RECONLENS_SYNTHETIC") == "1",
               data_manifest_sha256=cache.file_sha256(config.DATA_DIR / "MANIFEST.sha256") if (config.DATA_DIR / "MANIFEST.sha256").exists() else None,
               n_files=len(files), files=files)
    (config.RESULTS_DIR / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


if __name__ == "__main__":
    write_env()
    m = write_manifest()
    print(f"env.json written; manifest.json hashes {m['n_files']} files")
