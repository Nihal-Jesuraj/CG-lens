"""Single source of truth for ReconLens (plan v4.0). Every tunable lives here."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RECONLENS_DATA", ROOT / "data"))
RAW_DIR = Path(os.environ.get("RECONLENS_RAW", DATA_DIR / "raw"))
PROC_DIR = DATA_DIR / "processed"
CACHE_DIR = Path(os.environ.get("RECONLENS_CACHE", DATA_DIR / "cache"))
RESULTS_DIR = Path(os.environ.get("RECONLENS_RESULTS", ROOT / "results"))

# ---- reproducibility ------------------------------------------------------
GLOBAL_SEED = 2026
SNR_INF_CODE = 999  # integer seed key for SNR = infinity (never use hash())

# ---- image and operator settings ------------------------------------------
N = 128                 # reconstruction grid
N_FEAT = 64             # classification feature grid (128 -> 64 area average)
SYMMETRIC_MASK = True   # False only for the Section 4.2 (v) demonstration
CENTER_DISC_FRAC = 0.05  # fully sampled central disc radius = 0.05 * N
MASK_RATE_TOL = 0.01
R_LIST = [2, 4, 6, 8]
SNR_LIST = [math.inf, 25.0]
SNR_PRIMARY = 25.0
R_PRIMARY = 4
CAR_FRAC = 0.08         # 1-D Cartesian mask: central fraction fully sampled (P1)

# ---- reconstruction solver settings ---------------------------------------
CG_RTOL = 1e-6
CG_MAXIT = 200
IRLS_OUTER_TOL = 1e-4
IRLS_OUTER_MAX = 30
VAL_INNER_RTOL = 1e-10  # validation settings (E2b, E3)
VAL_OUTER_TOL = 1e-10
VAL_OUTER_MAX = 500
J_VIOLATION_TOL = 1e-12
NLCG = dict(c1=1e-4, shrink=0.5, restart=50, rtol=1e-4, maxit=300)

# ---- tuning grids ---------------------------------------------------------
M1_LAMBDAS = [10.0 ** e for e in (-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0)]
M2_LAMBDAS = [10.0 ** e for e in (-4, -3.5, -3, -2.5, -2, -1.5, -1)]
M2_MUS = [1e-2, 1e-3, 1e-4]
EDGE_MAX_EXTENSIONS = 2

# ---- data and cohorts -----------------------------------------------------
MIN_ROI_PIXELS = 30
DHASH_MAX_HAMMING = 6
# dHash is a candidate generator; retain an edge only when the min-max-scaled
# 128x128 images agree under a flip/rotation. See README deviation note.
DHASH_CONFIRM_RMSE = 0.05
DEDUP_SIZE = 128
MAX_SLICES_PER_PATIENT = 2
RECON_VAL_PATIENTS = 15
CLF_POOL_PATIENTS = 150
RECON_EVAL_PATIENTS = 40
E5_TEST_FRAC = 0.2
BINARY_LABELS = {1: -1, 2: +1}   # Cheng label -> SVM label (meningioma -1, glioma +1)

# ---- classification -------------------------------------------------------
PCA_VAR = 0.95
PCA_SENS_K = [10, 25, 50]
PCA_STANDARDIZE = False
TUNE_WITH_SKLEARN = True
C_GRID = [0.1, 1.0, 10.0, 100.0]
G_GRID = [0.1, 1.0, 10.0]
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
SVM_RIDGE = 1e-8
SMO_TOL = 1e-3
SV_THRESH = 1e-6

# ---- statistics -----------------------------------------------------------
BOOT_B = 10000
BOOT_B_EXPLORATORY = 2000   # exploratory Spearman CIs only (primary CIs use BOOT_B)
ALPHA = 0.05

# ---- E5 -------------------------------------------------------------------
E5_NS = [100, 250, 500, 1000, 1500]
E5_SEEDS = [0, 1, 2]
E5_C_GRID = [0.1, 1.0, 10.0, 100.0]
QP_TIME_CAP_S = 600
QP_DENSE_MAX_N = 500

# ---- tests and benchmark --------------------------------------------------
BENCH_SLICES = 10
BENCH_BUDGET_HOURS = 8.0
QUICK_SLICES = 16

LIMITATIONS = [
    "Synthetic k-space from already-reconstructed magnitude images; single coil, no phase, no scanner physics, single sequence.",
    "The same discrete model generates and inverts the data (inverse crime, Kaipio and Somersalo 2007); absolute PSNR/SSIM are optimistic.",
    "Downsampling to 128x128 removes fine detail; at most 2 slices per patient; one dataset, no external validation.",
    "The ROI-size rule (MIN_ROI_PIXELS) excludes small tumors from the reconstruction and classification cohorts; see roi_filter_audit.csv.",
    "The dataset mixes slice orientations and they are not labelled; a classifier could exploit orientation or position cues.",
    "Results show controlled algorithmic behaviour, not clinical performance.",
    "Deviation from PLAN_v4 §3 (conservative amendment): the plan's dHash Hamming distance <= 6 rule would remove 1,162 slices (54.5%) and leave a largest merged group of 314 slices from 34 patients. We use dHash <= 6 only to generate candidates, then confirm candidates with min-max-scaled 128x128 dihedral RMSE <= 0.05. The minimum cross-label RMSE among the 42,440 candidates is 0.089479. In 10,000 patient-label permutations (seed 2026), the null mean was 20,548.5 (SD 286.5; 95% range 19,823-20,949); 16,021 cross-label candidates were observed, below this range. Candidate counts alone do not establish duplication. The reproducible 12-pair cross-label sample has max-SSIM range 0.4299-0.5728; the exhaustive set of all 7 cross-label Hamming-0 pairs has range 0.4431-0.6158. The confirmed cross-label edge count is zero and the cross-class near-duplicate removal attrition row remains 0. Confirmation examines only dHash candidates (Hamming <= 6), so near-duplicates with larger Hamming distance are not caught.",
]

_HASHED = [
    "GLOBAL_SEED", "N", "N_FEAT", "SYMMETRIC_MASK", "CENTER_DISC_FRAC", "MASK_RATE_TOL",
    "R_LIST", "SNR_LIST", "CG_RTOL", "CG_MAXIT", "IRLS_OUTER_TOL", "IRLS_OUTER_MAX",
    "VAL_INNER_RTOL", "VAL_OUTER_TOL", "VAL_OUTER_MAX", "MIN_ROI_PIXELS", "DHASH_MAX_HAMMING",
    "MAX_SLICES_PER_PATIENT", "RECON_VAL_PATIENTS", "CLF_POOL_PATIENTS", "RECON_EVAL_PATIENTS",
    "DHASH_CONFIRM_RMSE",
    "PCA_VAR", "PCA_STANDARDIZE", "NLCG",
]


def snr_code(snr_db):
    """Integer seed key for an SNR value (never hash())."""
    return SNR_INF_CODE if not math.isfinite(snr_db) else int(round(snr_db))


def settings():
    g = globals()
    return {k: g[k] for k in _HASHED}


def _canon(o):
    if isinstance(o, float):
        return "inf" if math.isinf(o) and o > 0 else ("-inf" if math.isinf(o) else ("nan" if math.isnan(o) else o))
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item") and getattr(o, "shape", None) == ():
        return _canon(o.item())
    if hasattr(o, "tolist"):
        return _canon(o.tolist())
    return o


def canonical_json(obj):
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))


def config_hash():
    return hashlib.sha256(canonical_json(settings()).encode()).hexdigest()


def ensure_dirs():
    for d in (PROC_DIR, CACHE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
