# ReconLens

Code, tests and experiment scripts for plan v4.0 (course 22MAT220). This README records run commands, current data counts, and a documented deviation; the report text is written by the students.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-core.txt
```
Download the Cheng brain-tumor dataset (figshare DOI 10.6084/m9.figshare.1512427, version 8) into `data/raw/` (git-ignored).

`requirements-freeze.txt` preserves the local environment snapshot with machine-specific path entries removed. Use `requirements-core.txt` for the pinned direct dependencies.

## Current data counts

Counts below are from the current `results/attrition.csv` and `results/splits.json`:

- Raw data: 3,064 `.mat` files from 233 patients.
- Binary-label filter: 2,134 slices from 171 patients (glioma: 1,426 slices / 89 patients; meningioma: 708 / 82).
- ROI-eligible: 2,105 slices from 170 patients (glioma: 1,408 / 89; meningioma: 697 / 81).
- Cohort selection: 324 slices from 165 patients: recon_val has 30 slices / 15 patients and clf_pool has 294 / 150 (glioma: 172 / 86; meningioma: 152 / 79).

## Deviation from PLAN_v4 §3

PLAN_v4 §3 uses dHash Hamming distance ≤ 6, drops cross-class matches, and merges near-duplicates into groups. As a conservative amendment, dHash ≤ 6 is used to generate candidate pairs and a candidate is confirmed only if its min-max-scaled 128×128 dihedral RMSE is ≤ 0.05. The dHash-only rule would remove 1,162 slices (54.5%) and, after same-label merging, leave a largest group of 314 slices from 34 patients. It produces 16,021 observed cross-label candidates among 42,440 total candidates. In 10,000 patient-label permutations (seed 2026), the null mean was 20,548.5 (SD 286.5; 95% range 19,823–20,949); the observed count is below this range. Candidate counts alone do not establish duplication. The minimum cross-label RMSE among candidates is 0.089479. The reproducible 12-pair cross-label sample has max-SSIM range 0.4299–0.5728; the exhaustive set of all 7 cross-label Hamming-0 pairs has max-SSIM range 0.4431–0.6158. At RMSE ≤ 0.05 there are 9 same-label/same-PID edges, 0 same-label/different-PID edges, and 0 cross-label edges. The cross-class near-duplicate removal attrition row remains at 0. Confirmation only examines pairs already within dHash Hamming distance ≤ 6; near-duplicates with larger Hamming distance are not caught.

## Protocol notes

- E4 tuning consistency: `TUNE_WITH_SKLEARN` is set to false only when fewer than half of the folds are within one grid step. In the current audit, 4/5 folds were within one step; folds 0 and 4 disagreed on the selected sklearn and QP settings (fold 4 was still within one grid step).
- E4 protocol B's M0−M2 balanced-accuracy and macro-F1 bootstrap intervals have lower bounds exactly 0.000. These are bootstrap boundary cases; the “not significant” labels should not be read as a sharp distinction from neighboring “inconclusive” rows.
- E3 uses SNR 25 dB by default; pass `--snr inf` for the noise-free setting.
- E8/H5 at R=4 and SNR=25 dB: PSNR and SSIM fall on finer-grid data for M0, M1, and M2; method ordering M2 > M1 >= M0 is unchanged. See `results/tables/table_e8_crime.csv` for patient-clustered paired differences and 95% bootstrap CIs.
- History before 2026-11-23 is unavailable. The E1–E5 full pipeline run used source commit `e1cce40`. E3 validation-status code was changed in `6ac36d3`, with its generated validation outputs recorded in `af3da72`. The full E8 experiment used source commit `af3da72`; the H5 table generator and regenerated manifest were produced from `54a9642`, and the E8 outputs/table were recorded in `bfd724b`. No earlier milestone tags are claimed.
- `results/e5_log.txt` records solver skips/timeouts; it says “no skips or timeouts” when the run has none.

## Tests
```bash
python -m pytest -q                    # everything
python -m pytest -q -m "not slow"      # (no test is currently marked slow)
```

## Milestone-by-milestone (real data)
```bash
python -m src.data verify              # loader facts: shapes, dtypes, decoded PID of the first files
python -m src.data audit               # data_audit.json, attrition.csv, roi_filter_audit.csv  -> REVIEW THE COUNTS
python -m src.data cohorts             # splits.json
python -m src.data manifest            # data/MANIFEST.sha256
python scripts/benchmark.py            # G-BENCH: projected serial hours -> benchmark.json
python -m experiments.tune --n-jobs 4  # lambda, mu on recon_val only -> hyperparams.json (frozen)
python -m experiments.e1_recon --n-jobs 4
python -m experiments.e1_recon --timing           # serial wall-clock pass
python -m experiments.e3_cvx
python -m experiments.e2_cg [--p1]
python -m experiments.e4_diag --n-jobs 4
python -m experiments.e5_svm                      # serial; QP time cap in config.py
python -m experiments.e7_lambda                   # P1
python -m experiments.e8_crime                    # P1
python scripts/validation_summary.py
python scripts/make_figures.py
python scripts/make_tables.py
python scripts/write_env.py                       # env.json + manifest.json
```

## One command
```bash
python run_all.py --n-jobs 4                      # full run
python run_all.py --p1 --n-jobs 4                 # plus P1 items
python run_all.py --quick --synthetic             # smoke test on generated data (NOT scientific results)
python run_all.py --audit-only                    # stop after the dataset audit
```
Changing a frozen file (`hyperparams.json`) needs `python run_all.py --refreeze --reason "..."` and re-runs everything downstream.

## Limitations stated by the code (config.LIMITATIONS)
- Synthetic k-space from magnitude images; single coil; no phase or scanner physics.
- Inverse crime: same discrete model generates and inverts the data; absolute PSNR/SSIM are optimistic.
- 128x128 downsampling; at most 2 slices per patient; one dataset; no external validation.
- ROI-size rule (MIN_ROI_PIXELS = 30) excludes small tumors; counts are in `roi_filter_audit.csv`.
- Slice orientations are mixed and unlabelled.
- Controlled algorithmic behaviour, not clinical performance.
