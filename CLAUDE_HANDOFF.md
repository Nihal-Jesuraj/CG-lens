# ReconLens review handoff

Review this repository against `PLAN_v4.md`, checking implementation, protocol, and the current `results/` artifacts. Do not infer scientific claims from synthetic runs. This handoff contains Cheng-data outputs; assess them against the source tables and audit files before drawing conclusions.

## Reproducibility and data

- Install the pinned direct dependencies with `python -m pip install -r requirements-core.txt`. `requirements-freeze.txt` is a broad local snapshot; machine-specific `file://` entries were removed.
- Raw and processed MRI data are intentionally excluded from this archive. The README lists the Cheng dataset source and local verification commands.
- The project did not arrive with Git history. The README records: “No pre-submission history available.” Do not infer or invent milestone tags.
- E6, M11, and E11 remain excluded per PLAN_v4 Section 10.

## Method decisions to review

- PLAN_v4 §3 specifies dHash Hamming distance ≤6, cross-class removal, and near-duplicate grouping. The implementation uses dHash ≤6 to generate candidates and confirms them using min-max-scaled 128×128 dihedral RMSE ≤0.05, as a conservative amendment. The dHash-only rule would remove 1,162 slices (54.5%) and form a largest same-label group of 314 slices from 34 patients.
- The audit found 16,021 cross-label candidates among 42,440. A fixed-seed, 10,000-permutation patient-label null has mean 20,548.5, SD 286.5, and 95% range 19,823–20,949; the observed count falls below that range. Candidate counts alone do not establish duplication.
- SSIM evidence comes from two sets: the reproducible seeded 12-pair cross-label sample has max-SSIM range 0.4299–0.5728; the exhaustive set of all seven cross-label Hamming-0 pairs has range 0.4431–0.6158. See `results/dedup_audit_summary.json` and its source script.
- E4 tuning consistency is interpreted as false if fewer than half of folds are within one grid step. The current diagnostic reports 4/5 within one step; folds 0 and 4 disagree between sklearn and QP selected settings.
- E3 uses SNR 25 dB by default; `--snr inf` selects noise-free data. The NLCG test tolerance is 1e-7 because the function-value Armijo test cannot resolve 1e-9 in float64; review the test and comment.

## Where to look

- `results/manifest.json` lists and hashes included result files. `results/e1_examples.csv` is the Fig. 3 pixel table; `results/figures/fig03_examples.pdf` and `.png` are included when present.
- `results/e4_oof.csv` contains patient-disjoint out-of-fold predictions. Check its dataset identifiers and split provenance.
- `results/validation_summary.csv`, `results/e3_summary.csv`, and `results/e5_log.txt` expose solver validation, IRLS iteration-cap behavior, and E5 skips/timeouts.
- The README gives data counts and reproducible commands. The bundle excludes raw data, processed image arrays, caches, and the pre-final comparison copy.

Please report prioritized correctness, methodology, reproducibility, and plan-compliance issues with file/line references. Distinguish blocking fixes from optional improvements, and state which conclusions need additional evidence. Do not modify files or call the project complete.
