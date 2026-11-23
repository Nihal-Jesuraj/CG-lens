# ReconLens: Final Implementation Plan v4.0 (for DeepSeek V4.1)

Course 22MAT220 Mathematics for Computing-III · Theme A (Optimization & Machine Learning), application: undersampled MRI inverse problems · 22 Nov 2026 · supersedes v3.1 (every change is listed in Section 12)

**How the students use this file:** paste the whole file into the assistant once. At the start of every milestone, re-paste Section 0.5 (Conventions Card) plus that milestone's prompt from Section 8. Appendix S at the very end is for the students only; the assistant ignores it.

---

## 0. Rules for the coding assistant (read first)

You are helping two undergraduate students build a course project that may later become a conference paper. Your job is **code, tests and experiment scripts only**.

### 0.1 What you may and may not write

- **No submission prose.** Do NOT write the report, abstract, literature review, discussion, figure captions or any text meant for submission. The course enforces Turnitin (< 10%) and AI-similarity (< 20%) limits; the students write all prose themselves.
- Allowed: code, one-line technical docstrings, short comments, `config.py`, tests, and a README that contains **run commands only** (the students write the narrative).
- Give **complete files** with the path as a heading. No "..." elisions, no "rest unchanged". Prefer one module plus its tests per reply so nothing is truncated.

### 0.2 Execution protocol

- Assume you **cannot run code**. Give exact commands; the students run them and paste back the full terminal output. Never say a test passed, or quote a number, until it has been pasted. If you do have an execution tool, still show real output. Never fabricate output.
- Work **one milestone at a time** (Section 5). After the tests pass, explain every function in plain language (at most 8 lines each) so both students can defend it in the viva, then **wait for "next"**.
- At the start of each milestone, restate the Conventions Card (Section 0.5) in your own words in at most 15 lines, and flag anything in earlier files that violates it.

### 0.3 Honesty rule

- A failing **correctness test** means a bug: fix the bug. A test threshold may be changed only when the failure is diagnosed as a wrongly chosen numerical tolerance (not a bug), the new value is justified in an inline comment, and the change is listed in that milestone's summary.
- A failing **empirical hypothesis** (Section 5.2) or a result that contradicts expectation is a finding: print it, diagnose it, report it. Never adjust seeds, data, metrics or settings to produce an expected trend.
- **Freeze rule.** λ and μ are chosen on `recon_val` only and written once to `results/hyperparams.json`. (C, γ) are chosen by inner cross-validation on training folds only. Never tune on `recon_eval` or on outer test folds. Changing a frozen file afterwards needs a logged reason and a re-run of everything downstream.

### 0.4 No-invention rule

- Do not invent dataset facts, library APIs, version numbers, DOIs, or numerical results. If unsure, write "unverified" and add a small script that prints the fact (shapes, dtypes, counts, versions).
- `requirements.txt` is generated with `pip freeze` on the students' machine after the environment builds. Do not type pinned versions from memory.
- Ask at most 3 clarifying questions, and only if blocked. Otherwise state your assumption in one line and proceed.

### 0.5 Conventions Card (the invariants; re-paste at every milestone)

1. Image size N = 128 (tiny test problems use 16 or 32). Arrays are `float64`, shape (N, N). Each ground-truth slice is min-max scaled to [0, 1] once. Reconstructions are never rescaled, normalised or clipped (clipping only as an explicit ablation).
2. F = `np.fft.fft2(..., norm='ortho')`. Spectra are stored in unshifted FFT order; `fftshift` is used only for plotting.
3. Forward operator A x = M ⊙ (F x). Real inner product ⟨a, b⟩ = Re Σ conj(a)·b. Aᴴy = Re(F⁻¹(M ⊙ y)). AᴴA x = F⁻¹(M ⊙ F x), which is exactly real because M is conjugate-symmetric.
4. Mask M is conjugate-symmetric: M(−k) = M(k) with indices mod N. R = N² / m, counting both members of each conjugate pair. DC and a central disc are always sampled.
5. Noise: η = M ⊙ F(w), with w a real i.i.d. Gaussian image, rescaled so ‖η‖₂ = ‖A x_true‖₂ · 10^(−SNR_dB/20). This keeps η conjugate-symmetric, so nominal SNR is the SNR the reconstruction actually sees. SNR = ∞ means η = 0.
6. D stacks the **valid** forward differences with Neumann boundary: p = 2N(N−1). TV is **anisotropic**. Dᵀ is the negative divergence.
7. J(x) = ½‖Ax − y‖² + λ Ψ_μ(x), Ψ_μ(x) = Σ √((Dx)ᵢ² + μ²). IRLS is lagged diffusivity, warm-started, with μ fixed.
8. Labels: glioma = +1 (Cheng label 2), meningioma = −1 (Cheng label 1).
9. Features: 128×128 → 64×64 area average → flatten (4096). PCA centres only (no standardisation unless the `PCA_STANDARDIZE` flag is on), fitted on training data only, **refitted inside every inner and outer training set**.
10. SVM: class-balanced C_i = C · n_train / (2 · n_class(i)). RBF K = exp(−γ‖x − x′‖²) with γ = g / (n_features · Var(X_train_pca)), g ∈ {0.1, 1, 10}. Dual objective D(α) = ½ αᵀQα − Σα with Q = (yyᵀ) ∘ K, the same formula for every solver.
11. Seeds: `np.random.SeedSequence([GLOBAL_SEED, *integer_keys])`. Keys are plain integers (slice index, R, SNR code). **Never use Python `hash()`** (it is salted per process). Mask seed keys: (R, slice index). Noise seed keys: (R, SNR code, slice index). Every method sees identical data for a given (slice, R, SNR).
12. Metrics: PSNR with MAX = 1; SSIM from scikit-image with `data_range=1.0` and default window; NMSE = ‖x̂ − x‖² / ‖x‖²; ROI-NMSE = the same ratio over `tumorMask` pixels at 128×128.
13. Wall-clock timing only in a serial pass (`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`). Matvec counts (one A plus one Aᴴ) are the primary cost measure.

### 0.6 Libraries

- **Allowed:** `numpy`, `scipy` (FFT, linear algebra, `LinearOperator`, `eigsh` for diagnostics, `scipy.stats` including `wilcoxon`, `scipy.io` fallback; `scipy.sparse.linalg.cg` as a reference check only), `h5py`, `matplotlib`, `pandas`, `scikit-image` (metrics only), `scikit-learn` (baselines, `StratifiedGroupKFold`, evaluation metrics, `SVC` as reference and fast tuning evaluator only), `cvxopt`, `Pillow`, `joblib`, `pytest`.
- **Forbidden (outside the syllabus):** ADMM, ISTA/FISTA or any proximal-gradient method, wavelets, nuclear norm or singular value thresholding, PyTorch/TensorFlow or any deep learning.
- **Must be hand-written (course rule: core algorithms are ours):** linear CG, nonlinear CG, lagged-diffusivity IRLS, PCA via SVD, SVM in two versions (dual QP via CVXOPT and SMO), the difference-hash (dHash) duplicate detector with union-find grouping, and `stats.py` (Nadeau–Bengio corrected resampled t-test, patient-clustered bootstrap, Holm–Bonferroni). Library versions are used only for tests and baselines.

### 0.7 Engineering standards

- Python 3.10+, `float64` everywhere, one `config.py` as the single source of truth (including `GLOBAL_SEED = 2026`), git with one commit per milestone tagged `M1`, `M2`, …, `.gitignore` for `data/raw/` and the cache.
- **Caching:** every reconstruction is stored as `.npy`. The key is a SHA-256 of canonical JSON of (slice id, method, R, SNR, λ, μ, solver settings, mask seed keys, noise seed keys, config hash), so a changed setting can never reuse a stale file. Use `joblib` for parallel loops (never for timing).
- Save CPU model, core count, Python and library versions to `results/env.json`. Every run writes `results/manifest.json` with output file hashes.
- Priority tags: **P0** required for the course submission, **P1** if ahead of schedule, **P2** parked (Section 10).

---

## 1. Project summary

**Title:** *ReconLens: A Comparative Study of Optimization-Based Reconstruction and Patient-Disjoint SVM Classification for Undersampled Brain MRI*

**Central question:** How does optimization-based reconstruction of undersampled MRI k-space affect both image fidelity (whole slice and tumor region) and downstream patient-independent SVM tumor-type classification?

**Pipeline:** T1-CE brain MRI slice + radiologist tumor mask → simulate conjugate-symmetric undersampled noisy k-space → reconstruct with zero-filled (M0), Tikhonov-H¹ (M1) and smoothed-TV IRLS (M2) → measure PSNR, SSIM, NMSE, ROI-NMSE and solver cost → PCA features (fit on training patients only) → glioma vs meningioma with class-balanced hand-written SVMs (QP and SMO, checked against libsvm) under patient-disjoint cross-validation → test whether fidelity gains predict diagnosis gains.

**Positioning (for README and tables):** synthetic single-coil k-space from processed magnitude images; the study offers a transparent, mathematically verified classical baseline with patient-level leakage control. It does not establish clinical validity.

| Component | Role |
| --- | --- |
| Linear CG (hand-written) | Solves the SPD systems inside M1 and M2; its convergence explains solver cost |
| M1 Tikhonov-H¹, M2 TV-IRLS | The two competing reconstructions, M0 zero-filled as baseline |
| CVXOPT anisotropic TV QP | Exact constrained convex reference that validates M2 on 16×16 problems |
| PCA via SVD (hand-written) | Dimension reduction fitted on training patients only |
| SVM: dual QP and SMO vs libsvm | Class-balanced diagnosis and independent verification of the hand-written solvers |
| Patient-disjoint CV, clustered statistics | No slice-level patient leakage; valid inference across correlated slices |

**Research questions** (RQ3 is the headline question for any paper; RQ2 and RQ4 are course-report material)

- **RQ1:** How do M0, M1, M2 compare in PSNR, SSIM, NMSE, ROI-NMSE and matvec cost across R ∈ {2, 4, 6, 8} and SNR ∈ {∞, 25 dB}?
- **RQ2:** How does linear CG convergence depend on conditioning (μ, λ), and how does the energy-norm error compare with the bound ‖eₖ‖_A ≤ 2((√κ − 1)/(√κ + 1))ᵏ ‖e₀‖_A?
- **RQ3:** Does higher reconstruction fidelity translate into better patient-disjoint SVM classification across `original`, `original_noisy`, M0, M1, M2, under matched (Protocol A) and mismatched (Protocol B) training, and do per-slice fidelity gains track per-slice classification gains?
- **RQ4:** How do SVM training routes (CVXOPT dual QP, SMO, scikit-learn/libsvm) compare in dual objective, support-vector overlap, KKT gap, balanced accuracy and runtime scaling?

Scope is frozen: no new algorithms, no title changes. Stretch ideas live in Section 10.

---

## 2. Repository layout

```text
reconlens/
  README.md  requirements.txt  config.py  run_all.py  .gitignore
  data/
    raw/              # git-ignored Cheng .mat files (students download manually)
    MANIFEST.sha256   # generated by a script after download
  src/
    data.py           # loader, audit, dHash+union-find grouping, PID-grouped splits
    mri_ops.py        # masks, A, A^H, noise, D, D^T
    cg.py             # linear CG, nonlinear CG (P1), Armijo line search
    recon.py          # zero_filled, tikhonov_cg, tv_irls_cg, tv_nlcg (P1)
    metrics.py        # PSNR, SSIM, NMSE, ROI-NMSE
    qp_reference.py   # CVXOPT anisotropic TV QP
    pca.py            # PCA via economy SVD
    svm.py            # kernels, SVM_QP (CVXOPT), SVM_SMO
    stats.py          # corrected resampled t, clustered bootstrap, Holm
    cache.py          # keyed .npy cache
  tests/              # one pytest module per src file; slow tests marked
  experiments/        # e1_recon.py, e2_cg.py, e3_cvx.py, e4_diag.py, e5_svm.py, e7_lambda.py (P1), e8_crime.py (P1)
  scripts/            # benchmark.py, make_figures.py, make_tables.py
  results/            # csv, figures, tables, splits.json, folds.json, data_audit.json,
                      # hyperparams.json, benchmark.json, env.json, manifest.json (generated)
```

---

## 3. Data, intensity convention and leakage-free splitting (P0)

- **Dataset:** Jun Cheng brain tumor dataset, figshare (DOI 10.6084/m9.figshare.1512427, version 8). T1-weighted contrast-enhanced slices in MATLAB `.mat` files with fields `PID`, `label`, `image`, `tumorMask`. The students download it manually; the assistant must not assume network access. `MANIFEST.sha256` is produced by a script after the download.
- **Loading:** files are expected to be MATLAB v7.3 (HDF5), read with `h5py`; fall back to `scipy.io.loadmat` for older versions. HDF5 reads of MATLAB arrays typically come back **transposed** and strings as arrays of character codes. Do not assume: print shapes, dtypes and a decoded `PID` for the first few files, apply the same transposition to `image` and `tumorMask`, and write the verified facts into `results/data_audit.json`.
- **Task:** binary glioma (label 2) vs meningioma (label 1). The dataset description gives about 1,426 glioma and 708 meningioma slices; the audit must print the real counts and flag any difference. Record **both patient counts and slice counts per class**.
- **Audit (`results/data_audit.json`):** slice and patient counts per class; any `PID` with more than one label (flag and exclude); histogram of original image sizes (**do not assume 512×512**); dtype and intensity ranges; number of empty masks; mask area at 128×128.
- **Attrition table (`results/attrition.csv`):** slices and patients per class after every stage: raw files → binary label filter → label-consistent `PID` → exact-duplicate removal → cross-class near-duplicate removal → constant-image removal → ROI-eligible → cohort selection. The counts quoted from the dataset description are not targets; no stage may be adjusted to reach them, and the README quotes the final counts from this table.
- **Preprocessing:** area-average (box) downsampling to 128×128 in `float64` (Pillow `BOX` or block averaging; no augmentation); `tumorMask` area-averaged and binarised at > 0.5. Per-slice min-max scaling of the ground truth to [0, 1]; drop slices with a constant image.
- **Slice eligibility:** a slice is eligible for the reconstruction and classification cohorts only if its 128×128 mask has at least `MIN_ROI_PIXELS = 30` pixels (fixed now, before any result is seen).
- **ROI-filter audit (`results/roi_filter_audit.csv`):** per class, the number of slices excluded; the number of patients affected, and how many of them lose all their slices and so drop out of the cohorts; and tumor-mask area quantiles (min, 5th, 25th, 50th, 75th, 95th percentile, max) before and after the filter. The rule removes small tumors, so this selection effect must be visible in the results.
- **Duplicates and groups:** exact SHA-256 plus hand-written 64-bit dHash (Hamming distance ≤ 6, also checking flips and rotations). Drop duplicates that cross class labels. Merge near-duplicate slices and `PID` into connected components with a hand-written union-find. **No group may cross a train, validation or test boundary.**
- **Intensity convention:** only the ground truth is min-max scaled. `original_noisy`, M0, M1, M2 stay in the raw solver scale for metrics and features.
- **Cohorts (saved once to `results/splits.json`; sizes in `config.py`):** at most 2 slices per patient, chosen by a seeded uniform draw among that patient's eligible slices.
  - `recon_val`: 15 patients (about 30 slices), class-stratified, used only for tuning λ and μ.
  - `clf_pool`: 150 patients (about 300 slices), disjoint from `recon_val`, used for E4. If the audit shows fewer than 165 eligible patients, shrink `clf_pool` (keep `recon_val` at 15) and record it.
  - `recon_eval`: 40 patients (about 80 slices), a fixed subset of `clf_pool`, used for E1 tables, ROI analysis and error maps (not a classification test set).
  - `E5_pool`: all binary slices that survive the label-consistency, duplicate and constant-image filters (the ROI-size rule is not applied, since E5 uses no ROI metric), one 80/20 patient-disjoint split, no reconstruction.
- **Limitations that code, README and results tables must state (students phrase the report text):**
  - Synthetic k-space from already-reconstructed magnitude images; single coil, no phase, no scanner physics, single sequence.
  - The same discrete model generates and inverts the data (an "inverse crime", Kaipio and Somersalo 2007); absolute PSNR/SSIM are optimistic.
  - Downsampling to 128×128 removes fine detail; at most 2 slices per patient; one dataset, no external validation.
  - The ROI-size rule excludes small tumors from the reconstruction and classification cohorts, so results do not cover them; exclusion counts per class come from the ROI-filter audit.
  - The dataset mixes slice orientations and they are not labelled, so a classifier could exploit orientation or position cues instead of tumor appearance.
  - Results show controlled algorithmic behaviour, not clinical performance.

---

## 4. Mathematical specification

### 4.1 Forward model and noise

- x ∈ ℝ^(N×N), N = 128, F unitary 2-D DFT. A x = M ⊙ (F x); y = A x_true + η.
- Real inner product ⟨a, b⟩ = Re Σ conj(a)·b. Aᴴy = Re(F⁻¹(M ⊙ y)); AᴴA x = F⁻¹(M ⊙ F x), exactly real and symmetric positive semidefinite.
- Noise: η = M ⊙ F(w), w real Gaussian image, scaled to the target norm (Convention 5).
- **Tests:** |⟨Ax, y⟩ − ⟨x, Aᴴy⟩| < 1e-10 for random real x and complex y; the imaginary part of F⁻¹η is below 1e-12; ‖η‖₂ matches the nominal value to 1e-12; identical seed keys give identical masks and noise; different keys give different ones.

### 4.2 Masks

- **Primary (P0): 2-D variable-density random, conjugate-symmetric.** Signed frequency indices kᵢ ∈ [−N/2, N/2); normalised radius r(k) = ‖k‖ / (N/2), clipped to ≤ 1. p(k) = clip(a·(1 − r)³, 0, 1). Force DC and a fully sampled central disc of radius 0.05·N; find a by bisection so that the mean of p over the grid equals 1/R.
- **Symmetry recipe:** the partner of FFT index (i, j) is ((−i) mod N, (−j) mod N): `partner(M) = np.roll(np.flip(M, axis=(0, 1)), shift=(1, 1), axis=(0, 1))`. Build `pair_id` = the smaller of a point's linear index and its partner's. Self-conjugate points (DC and the Nyquist row and column) form single-point pairs and count once in m; all other pairs count twice. Draw one uniform number per `pair_id`; M = (u[pair_id] < p).
- **Exact rate:** a plain Bernoulli draw often misses the target by more than 3%. After drawing, add or remove whole pairs outside the central disc, uniformly at random with the seeded generator, until m is within 1% of round(N²/R).
- **Tests:** (i) M == partner(M) exactly; (ii) achieved R within 3% of target for R ∈ {2, 4, 6, 8} over 20 seeds; (iii) imaginary part of F⁻¹(M ⊙ F x) below 1e-12 for random real x; (iv) at N = 16, the dense H = AᴴA built from unit vectors satisfies H·H = H to 1e-10 and trace(H) = m; (v) demonstration, not pass/fail: with `SYMMETRIC_MASK = False`, H is not a projection (half-weight eigenvalues appear); print both eigenvalue sets.
- R = 1 (full mask) is used for the `original` and `original_noisy` rows. **Secondary (P1):** 1-D Cartesian phase-encode lines with the central 8% fully sampled, in conjugate-symmetric pairs.

### 4.3 Total variation, smoothing, majorization

- D = [Dx; Dy], slicing-based valid differences: Dx has N(N−1) rows, Dy has N(N−1) rows, p = 2N(N−1). No identically-zero boundary rows (they would add a constant 2Nμ to Ψ_μ). Dot-product test for D and Dᵀ.
- Anisotropic TV on purpose: Ψ₀(x) = Σ|(Dx)ᵢ| gives an exact QP for the CVXOPT reference, and IRLS must use the same form.
- φ_μ(t) = √(t² + μ²); Ψ_μ(x) = Σ φ_μ((Dx)ᵢ). Gap bound: 0 ≤ Ψ_μ(x) − Ψ₀(x) ≤ pμ for all x.
- ∇J(x) = Aᴴ(Ax − y) + λ Dᵀ(W(x) D x), W(x) = diag(((Dx)ᵢ² + μ²)^(−1/2)).
- **Majorize-minimize:** ψ(s) = √(s + μ²) is concave in s = t², so Q(x; x_k) = ½‖Ax − y‖² + (λ/2) Σ wᵢ (Dx)ᵢ² + const, with wᵢ = 1/√((Dx_k)ᵢ² + μ²), satisfies Q(x; x_k) ≥ J(x) and Q(x_k; x_k) = J(x_k).
- **Test:** finite-difference gradient check, relative error below 1e-5, on a 16×16 image.

### 4.4 Reconstruction methods and tuning

- **M0 zero-filled (P0):** x = Aᴴy.
- **M1 Tikhonov-H¹ (P0):** solve (AᴴA + λ DᵀD) x = Aᴴy by linear CG. SPD for λ > 0 because ker(DᵀD) = span{1} and the DC frequency is always sampled. (An identity penalty would give only two distinct eigenvalues and a scaled zero-filled image.)
- **M2 TV by lagged-diffusivity IRLS plus CG (P0):** x₀ = Aᴴy. For k = 0, 1, …: W_k = W(x_k); solve (AᴴA + λ Dᵀ W_k D) x_{k+1} = Aᴴy by CG **warm-started at x_k**, μ fixed.
  - Default: inner CG rtol 1e-6, `maxit` 200; outer stop when ‖x_{k+1} − x_k‖ / ‖x_k‖ < 1e-4 or after 30 outer iterations.
  - Validation (E2b, E3): inner rtol 1e-10, outer tolerance 1e-10, up to 500 outer iterations.
  - **Monitors (never hidden):** log J at every outer iteration and count violations of J_{k+1} − J_k > 1e-12·max(1, |J_k|); log inner iteration counts and the number of inner solves that hit `maxit` without reaching rtol.
- **M3 TV by nonlinear CG (P1):** Polak–Ribière+ (β = max(0, β_PR)), Armijo backtracking (c = 1e-4, shrink 0.5), restart every 50 iterations, rtol 1e-4, `maxit` 300.
- **Hyperparameter tuning (new):** each (method, R, SNR) cell gets its own parameters, chosen on `recon_val` by mean PSNR.
  - M1: λ ∈ logspace(−4, 0, 9). M2: λ ∈ logspace(−4, −1, 7) and μ ∈ {1e-2, 1e-3, 1e-4}.
  - **Edge rule:** if the best λ lies on a grid edge, extend the grid by one decade on that side (at most 2 extensions), evaluate only the new values, and log it. Store the full PSNR-versus-parameter table (it feeds E2e) and the final choices in `results/hyperparams.json`, frozen before `recon_eval` is touched.

### 4.5 Hand-written linear CG

- Standard recurrences for an SPD `matvec`, right-hand side b, x₀, rtol 1e-6, `maxit` 200, optional `callback(k, x_k)`. Returns x, iterations, matvec count, residual-norm history, a converged flag.
- **Curvature check (positive definiteness only):** raise `NonSPDError` if pᵀHp ≤ 0. This does **not** detect asymmetry: a non-symmetric H can still give pᵀHp > 0.
- **Symmetry check (separate):** provide `check_symmetry(matvec, n, trials=5)`: for random real u, v verify |uᵀ(Hv) − vᵀ(Hu)| ≤ 1e-10·(|uᵀ(Hv)| + |vᵀ(Hu)|). Use it in tests and call it once for every operator (the M1 system, the IRLS system at a sample W, the 32×32 study operator) before it is handed to CG in an experiment.
- Bound ‖eₖ‖_A ≤ 2((√κ − 1)/(√κ + 1))ᵏ ‖e₀‖_A with ‖v‖_A = √(vᵀHv). Plot it **only** on A-norm error curves (dense κ at 32×32; reference CG at rtol 1e-12 for x* at 128×128), never on 2-norm residual plots.

### 4.6 CVXOPT anisotropic TV reference (P0, 16×16)

- Real-stacked A_r = [Re(A); Im(A)] ∈ ℝ^(2m×n), y_r = [Re(y); Im(y)]. Test ‖A_rᵀA_r − H‖_F < 1e-10.
- Solve min over (x, t): ½‖A_r x − y_r‖² + λ 1ᵀt subject to −t ≤ Dx ≤ t, with `cvxopt.solvers.qp`: P = blockdiag(A_rᵀA_r, 0) + 1e-10·I, q = (−A_rᵀy_r; λ·1), G = [[D, −I], [−D, −I]] built as an `spmatrix`, h = 0. Variables are ordered (x, t). Convert every array to `cvxopt.matrix`; turn off progress output.
- **Tolerance ladder:** start from default tolerances, then tighten (1e-8, 1e-9, 1e-10) and keep the tightest setting that returns status `optimal`. Record status, tolerances, iterations and time per problem. Document the 1e-10 ridge and bound its effect on the objective.

### 4.7 Hand-written PCA

Centre with the training mean only; economy SVD X_c = UΣVᵀ; smallest k with ≥ 95% explained variance (also log k ∈ {10, 25, 50} as sensitivity). Test singular values, subspace principal angles (cos θ ≥ 1 − 1e-8) and reconstruction error against `sklearn.decomposition.PCA`; test that refitting after changing the test data leaves components unchanged. Zero-variance guard if `PCA_STANDARDIZE` is on.

### 4.8 Class-balanced SVMs (P0)

Labels ±1, linear and RBF kernels, per-sample bounds C_i (Convention 10; matches `SVC(class_weight='balanced')`). Precompute the kernel matrix.

- **Solver A, `SVM_QP`:** `cvxopt.solvers.qp` with P = (yyᵀ)∘K + 1e-8·I, q = −1, G = [−I; I] as an `spmatrix`, h = [0; C_vec], A = yᵀ (1×n, type `'d'`), b = 0. Support vectors are α_i > 1e-6·C_i. The bias b is the mean of y_i − Σ_j α_j y_j K_ij over free vectors (1e-6·C_i < α_i < (1 − 1e-6)·C_i); if none are free, use the midpoint of the KKT bounds.
- **Solver B, `SVM_SMO` (changed from v3.1):** SMO with **maximal-violating-pair selection**, deterministic. Maintain the gradient g = Qα − 1. I_up = {t: (y_t = +1 and α_t < C_t) or (y_t = −1 and α_t > 0)}; I_low = {t: (y_t = −1 and α_t < C_t) or (y_t = +1 and α_t > 0)}. Pick i = argmax over I_up of (−y_t g_t) and j = argmin over I_low of (−y_t g_t). Stop when the gap (−y_i g_i) − (−y_j g_j) ≤ `tol = 1e-3` (the same criterion as libsvm). Otherwise do the two-variable update that preserves yᵀα = 0 and respects the per-sample boxes: derive the feasible segment [L, H] for α_j from C_i, C_j and the labels. Let η = K_ii + K_jj − 2K_ij (≥ 0 for a PSD kernel). If η > 1e-12·(K_ii + K_jj), take the unconstrained minimiser along the pair direction and clip it to [L, H]. If η ≤ 1e-12·(K_ii + K_jj) (for example two identical kernel rows), the dual is linear along the pair direction, so evaluate the dual objective at both endpoints L and H and keep the lower one (as in Platt 1998). Do not replace a zero curvature by an arbitrary small floor. Then update g from kernel rows i and j. Bias: mean of (−y_t g_t) over free vectors, else the midpoint of the final bounds. Log the final gap, iteration count and maximum KKT violation.
- **Comparison with `SVC`:** build α for `SVC` by scattering |`dual_coef_`| into a length-n vector, then evaluate D(α) with the same formula for all three solvers.

---

## 5. Milestones: correctness gates vs research hypotheses

Correctness tests must pass before moving on. Empirical hypotheses (5.2) are scientific questions reported honestly whatever the outcome.

### 5.1 Correctness gates

| ID | Pri | Deliverable | Gate (must pass) |
| --- | --- | --- | --- |
| M1 | P0 | `config.py`, `data.py`, `mri_ops.py`, `metrics.py`, `cache.py` | Loader verified; audit, attrition table and ROI-filter audit written; dHash/union-find groups built; `splits.json` has zero patient overlap between `recon_val` and `clf_pool`; all Section 4.1 and 4.2 tests; D dot-product test; ROI-NMSE uses the resized mask and the `MIN_ROI_PIXELS` rule; cache key changes with any setting; seed-key determinism tests |
| M2 | P0 | `cg.py`, M0, M1 in `recon.py` | CG matches `np.linalg.solve` to 1e-8 on 16×16; iteration count matches `scipy.sparse.linalg.cg`; finite termination (≤ n steps on SPD, ≤ 3 steps with 3 distinct eigenvalues); one step for H = cI; `NonSPDError` fires on an indefinite matrix; `check_symmetry` rejects H = S + K with S SPD and K a small skew-symmetric part (pᵀHp stays positive, so only the symmetry check can catch it) |
| M3 | P0 | M2 (TV-IRLS), J, ∇J | Gradient check < 1e-5; J non-increasing at fixed μ within tolerance (violation count logged) |
| **G-BENCH** | P0 | `scripts/benchmark.py` | After M3: time 10 slices × {M0, M1, M2}, extrapolate the total serial hours of tuning + E1 + E4 reconstructions, write `results/benchmark.json`; if the projection exceeds 8 h, apply the cut order (5.3) before continuing |
| M4 | P0 | `qp_reference.py` | A_rᵀA_r = H; CVXOPT status optimal on the tolerance ladder; gap bound 0 ≤ J_μ(x_IRLS) − J₀(x_IRLS) ≤ λpμ at every μ |
| M5 | P0 | `stats.py`, tuning, E1, E2(a, b), E3 | `stats.py` reproduces hand-computed Nadeau–Bengio, clustered-bootstrap and Holm examples to 1e-10; `hyperparams.json` frozen; E1, E2, E3 CSVs and figures written |
| M6 | P0 | `pca.py`, `svm.py` (QP + SMO) | PCA matches sklearn (Section 4.7); both SVMs satisfy yᵀα = 0 and 0 ≤ α_i ≤ C_i; SMO final gap ≤ tol; dual objective within 1e-3 (QP) and 1e-2 (SMO) of `SVC` (report the actual values); kernel PSD check; toy separable set has the known hard-margin solution; a duplicate-row test (identical training points, zero curvature for that pair) neither crashes nor increases the dual objective, for QP and SMO |
| M7 | P0 | E4 and E5 | Identical `folds.json` for every condition; zero patient overlap in every fold; one OOF prediction per `clf_pool` slice per repeat; tuning-consistency and solver-agreement diagnostics logged |
| M8 | P0 | `run_all.py`, README, `requirements.txt`, `env.json`, `manifest.json`, `make_figures.py`, `make_tables.py` | One command reproduces everything; `run_all.py --quick` (16 slices, R = 4) finishes in under 10 min; full `pytest` passes; every figure and table is produced by a script from CSVs |
| M9 | P1 | M3 nonlinear CG, E2(c–e), 1-D mask | Gradient check < 1e-5; Armijo descent verified |
| M10 | P1 | E7, E8 | Discrepancy root within bisection tolerance; inverse-crime comparison table saved |

### 5.2 Empirical hypotheses

- **H1 (fidelity):** M1 and M2 improve mean PSNR, SSIM, NMSE and ROI-NMSE over M0 on `recon_eval`; M2 outperforms M1 on piecewise-smooth structure at R ≥ 4.
- **H2 (TV approximation):** on 16×16 slices, the IRLS-to-QP relative difference and J₀ gap shrink as μ goes from 1e-1 to 1e-5, until the floor set by the QP ridge and IRLS tolerances.
- **H3 (diagnosis):** M1 and M2 recover balanced accuracy relative to M0 under Protocols A and B, and per-slice fidelity gains correlate positively with per-slice margin gains.
- **H4 (solver equivalence):** SVM_QP and SVM_SMO reach ≥ 98% held-out prediction agreement with `SVC` at R = 4; remaining differences are explained by stopping tolerances.
- **H5 (P1, inverse crime):** absolute PSNR/SSIM fall when data come from a finer grid (E8), but the ordering of methods is unchanged.

### 5.3 Floor and cut order

**Floor (the minimum that still satisfies all four Theme A bullets):** M1–M6; E1 at R ∈ {4, 8}; E2(a1); E3; E4 at R = 4, SNR = 25 dB with Protocols A and B and 1 repeat; E5 up to n = 500.

**Cut order if time runs short:** M10 → M9 → E2(c–e) → QP runs at n ≥ 1000 in E5 → secondary SNR in E4 → R = 2 and R = 6 in E4 → repeats 3 → 1. Never cut M4, E3, scikit-learn validation tests, `PID`-grouped splitting, or paired statistics.

---

## 6. Experiments and statistical protocol

### E1 (P0) Reconstruction grid

M0, M1, M2 × R ∈ {2, 4, 6, 8} × SNR ∈ {∞, 25 dB} on the 80 `recon_eval` slices, with the frozen per-cell parameters. Report PSNR, SSIM, NMSE, ROI-NMSE, matvecs and serial wall time. Aggregate each metric per patient first (n = 40 pairs), then paired Wilcoxon signed-rank, 10,000-sample patient-level bootstrap 95% CIs, Holm correction.
**Primary family (declared in advance):** PSNR and ROI-NMSE for M1 vs M0, M2 vs M0, M2 vs M1 at R = 4, SNR = 25 dB (6 tests, Holm). Everything else is exploratory and labelled so.

### E2 (P0/P1) Solver convergence

(a1) 32×32 theory-matched CG study: ‖eₖ‖_A / ‖e₀‖_A against the κ bound for μ ∈ {1e-1, 1e-2, 1e-3, 1e-4}, with an eigenvalue histogram. (a2) 128×128 empirical residual and energy-norm error. (b) IRLS J_μ(x_k) descent with warm-started inner CG iterations and the monotonicity violation count. (c–e, P1) NLCG descent, NMSE versus matvecs, (λ, μ) sensitivity from the tuning tables.

### E3 (P0) CVXOPT exact-TV validation

10 slices from `recon_val`, area-averaged to 16×16 (n = 256, p = 480), R ∈ {2, 4}, μ ∈ {1e-1, …, 1e-5}, λ ∈ {1e-3, 1e-2, 1e-1} (the 128×128 tuned values do not transfer to a different grid size). IRLS at validation settings vs CVXOPT QP. Report relative solution difference, J₀ gap, gap-bound check, duality gap and solver status, tolerance and timing.

### E4 (P0) Patient-disjoint reconstruction-to-diagnosis

- **Feature sources:** `original` (R = 1, noise-free), `original_noisy` (R = 1, 25 dB; separates noise from undersampling), and M0, M1, M2 × R ∈ {2, 4, 6, 8} × SNR ∈ {∞, 25 dB}. All `clf_pool` slices are reconstructed once and cached.
- **Pipeline:** 128 → 64×64 area average → PCA (95% variance, hand-written) → class-balanced `SVM_QP` (RBF primary, linear secondary). A majority-class reference (balanced accuracy 0.5) is reported alongside.
- **Protocol A (matched):** train and test on the same reconstructed condition. **Protocol B (mismatched):** train on `original` (PCA and scaling fitted on `original` training data), then project the held-out reconstructions onto that PCA basis and test.
- **Folds:** outer 5-fold `StratifiedGroupKFold` (groups = `PID` plus duplicate components) × 3 repeats (J = 15), generated once, saved to `results/folds.json`, **identical for every condition, protocol and model** (so all comparisons are paired). Inner 3-fold `StratifiedGroupKFold` tunes C ∈ {0.1, 1, 10, 100} and g ∈ {0.1, 1, 10}, **refitting PCA inside each inner training fold**.
- **Tuning shortcut (`TUNE_WITH_SKLEARN = True`):** the inner search uses `SVC` as a fast evaluator; the selected (C, γ) then train the hand-written `SVM_QP` on the whole outer-training fold. Two checks are mandatory: (1) **tuning-consistency**: on `original`, one repeat, repeat the inner search with `SVM_QP`; report how many folds agree exactly and how many are within one grid step; if most disagree, set the flag to False and record it; (2) **solver agreement** at R = 4 (both SNR, both protocols): train `SVM_QP`, `SVM_SMO` and `SVC` with the same (C, γ) on the same folds; report prediction agreement (target ≥ 98%) and the balanced-accuracy difference.
- **Saved OOF table:** repeat, fold, slice id, PID, condition, protocol, true label, decision value, signed margin y·f(x), prediction.
- **Metrics:** balanced accuracy (primary), macro-F1, sensitivity and specificity (glioma = positive), ROC-AUC, confusion matrices summed over outer test folds. Mean ± std over the J resamples.
- **Inference (two procedures, never conflated):**
  1. **Fold-level, Nadeau–Bengio corrected resampled t-test:** paired fold differences d_j (j = 1..15), variance (1/J + n_test/n_train)·s_d², t distribution with J − 1 = 14 degrees of freedom. These give the **p-values** (approximate for repeated k-fold, so indicative).
  2. **Patient-clustered bootstrap (B = 10,000):** resample patients with replacement and keep all their OOF slices; each slice's correctness is averaged over the 3 repeats; balanced accuracy is the mean of per-class mean correctness. These give the **95% CIs** for the same paired differences.
  - **Primary family (declared in advance):** balanced accuracy and macro-F1 at R = 4, SNR = 25 dB, Protocols A and B separately, all 6 pairwise comparisons among {original, M0, M1, M2}, Holm within each (metric, protocol) family.
  - **Claim rule:** call a difference significant only if the Holm-adjusted p < 0.05 **and** the bootstrap CI excludes 0; if the two disagree, report it as inconclusive.
  - **Exploratory association:** per (method, R, SNR) cell, Spearman ρ between the per-slice improvement in ROI-NMSE (vs M0) and the change in signed margin (vs M0), with a patient-clustered bootstrap CI; the change in correctness is reported as a secondary outcome.
- Expect wide intervals (about 150 patients); report that plainly.

### E5 (P0) SVM solver comparison

On `E5_pool` features (no reconstruction; PCA fitted on each training subset), n ∈ {100, 250, 500, 1000, 1500}, 3 seeds, class-balanced `SVM_QP`, `SVM_SMO`, `SVC`, C = 1 and g = 1 for the scaling study. Report serial solve time (kernel build time separately), dual objective, support-vector count and Jaccard overlap, KKT gap, accuracy and balanced accuracy; linear vs RBF; accuracy vs C ∈ {0.1, 1, 10, 100} at n = 500.
**QP cost control:** the dense QP is mandatory for n ≤ 500. For larger n, extrapolate the n = 500 time (roughly ∝ n³) and run only if the prediction is below `QP_TIME_CAP_S` (default 600 s), in a separate process with a hard timeout; otherwise log "skipped: predicted above cap". SMO and `SVC` run at every n.

### E7 (P1) Noise-adaptive λ

Estimate the noise level from sampled entries in the outer 10% of k-space; choose λ by bisection on log λ so that ‖Ax_λ − y‖₂ matches the expected noise norm (discrepancy principle). Compare with the fixed tuned λ and the oracle λ (best PSNR). Theory link: the constrained form (minimise Ψ subject to ‖Ax − y‖ ≤ ε) is the penalised form with λ as Lagrange multiplier.

### E8 (P1) Inverse-crime sensitivity

On `recon_eval`, generate k-space from a 256×256 area-averaged version of each slice, keep the central 128×128 block of k-space (rescaled so mean intensity matches; test with a noise-free fully sampled case), apply the mask there, and reconstruct on the 128×128 grid against the area-averaged 128×128 ground truth. Report the PSNR/SSIM change against the matched-model results for M0, M1, M2 at R = 4. Tuned parameters are not re-tuned.

---

## 7. Viva derivations, figures, tables, paper-ready outputs

**Hand derivations (both students):**
1. Tikhonov normal equations and why the matrix is SPD (the DC term must be sampled).
2. Gradient of the smoothed TV objective and the weight matrix W.
3. IRLS as majorize-minimize: tangent bound for √(s + μ²) in s = t², the chain J(x_{k+1}) ≤ Q(x_{k+1}; x_k) ≤ Q(x_k; x_k) = J(x_k), why a fixed point has ∇J = 0, why warm-started inexact CG keeps descent.
4. CG as Krylov minimisation, conjugacy relations, the Chebyshev energy-norm bound, why it is not a residual bound, finite termination.
5. Class-weighted SVM Lagrangian, dual, KKT cases (α = 0, 0 < α < C_i, α = C_i).
6. The exact CVXOPT matrices for the SVM and the TV problem (how |·| becomes −t ≤ Dx ≤ t) and the gap bound 0 ≤ Ψ_μ − Ψ₀ ≤ pμ.
7. Why PCA by SVD gives the variance-maximising directions.
8. Why the adjoint contains Re(·), and what a conjugate-symmetric mask does to AᴴA.
9. The maximal-violating-pair rule, the two-variable SMO update (including the zero-curvature endpoint case), and why the gap m − M is a valid stopping test.
10. (P1) Discrepancy principle and λ as a Lagrange multiplier.

**Figures:** (1) pipeline; (2) masks at each R; (3) example reconstructions with whole-slice and tumor error maps; (4) PSNR, SSIM, ROI-NMSE vs R with CIs; (5) CG energy-norm error with κ bound (32×32) and empirical errors (128×128); (6) IRLS J descent and inner iterations; (7) IRLS-to-QP error vs μ; (8) PCA explained variance; (9) balanced accuracy vs R for Protocols A and B with clustered CIs; (10) SVM time and support vectors vs n; (11) confusion matrices for the primary comparison; (12, exploratory) fidelity-gain vs margin-gain scatter.
**Tables:** dataset audit with the attrition table (patients and slices per class at each stage) and the ROI-filter audit; reconstruction metrics with Wilcoxon/Holm; solver validation summary (CG vs dense and theory, IRLS vs CVXOPT, PCA vs sklearn, QP and SMO vs libsvm with achieved agreement); primary E4 table; chosen hyperparameters; limitations.

**Paper-ready outputs (M8):** `make_figures.py` writes each figure as vector PDF plus 300-dpi PNG with fixed matplotlib settings; `make_tables.py` writes CSV and LaTeX (`.tex`) for every table; both read only from `results/` CSVs (no manual editing); `results/headline.csv` lists every primary-endpoint number; the manifest records hashes so any figure can be traced to its inputs.

---

## 8. Milestone prompts (paste one at a time, with Section 0.5)

**Template:** "Plan v4.0 and the Conventions Card are above. Implement Milestone Mx only: [files]. Meet the gate in Section 5.1 row Mx. Give complete files, the exact commands to run, and what output I should paste back. Do not write any report text. Do not claim a test passed until I paste the output. When it passes, explain each function in at most 8 lines and wait for me to say 'next'."

- **M1:** config.py, src/data.py, src/mri_ops.py, src/metrics.py, src/cache.py and their tests. Use Section 3 and Sections 4.1–4.3 exactly: verify the `.mat` loader (transposition, PID decoding) and write the audit, the attrition table and the ROI-filter audit first, and stop to show me those counts before building cohorts; dHash plus union-find groups; cohorts and `splits.json`; unitary FFT with real inner product; conjugate-symmetric masks with exact-rate pair correction and the N = 16 projection test; mirrored-noise generation; anisotropic Neumann D with p = 2N(N−1); ROI-NMSE with `MIN_ROI_PIXELS`; SHA-256 cache; seed keys without `hash()`.
- **M2:** src/cg.py (linear CG, `NonSPDError`, and a separate `check_symmetry`) and M0, M1 in src/recon.py, with tests from the M2 gate.
- **M3:** M2 (TV-IRLS) in src/recon.py, J and ∇J, monotonicity monitor and inner-convergence counters. Then `scripts/benchmark.py` (G-BENCH).
- **M4:** src/qp_reference.py with the tolerance ladder, sparse G and all gate tests.
- **M5:** src/stats.py with hand-computed test cases first; then the tuning script with the edge rule and `hyperparams.json`; then experiments E1, E2(a, b), E3.
- **M6:** src/pca.py and src/svm.py (SVM_QP, then SVM_SMO with maximal-violating-pair selection) with the M6 gate tests.
- **M7:** experiments/e4_diag.py (generate `folds.json` first, then the two mandatory checks, then the grid) and experiments/e5_svm.py with the QP time cap.
- **M8:** run_all.py (with `--quick`), README run commands, env and manifest writers, make_figures.py, make_tables.py.

---

## 9. Common pitfalls

- Forgetting Re(·) in the adjoint (breaks CG symmetry); a non-symmetric mask; noise that is not conjugate-symmetric.
- Periodic instead of Neumann boundary in D; including zero boundary differences in Ψ; mixing isotropic and anisotropic TV between IRLS and the QP.
- Running CG on a non-SPD system; cold-starting the inner CG; comparing J across different μ; drawing the A-norm bound on a residual plot.
- Running the E3 comparison at the default 30 outer iterations instead of the validation settings.
- HDF5 transposition and character-code `PID` strings from MATLAB v7.3 files; assuming every image is 512×512.
- Python `hash()` in seeds; different fold splits for different conditions (breaks pairing); parallel timing; stale cache after a config change.
- Fitting PCA or scaling on data that includes test patients (also inside inner folds); tuning on `recon_eval`; best λ on the grid edge without applying the edge rule; using augmented or near-duplicate slices across train and test.
- A naive paired t-test or Wilcoxon over CV folds; claiming significance outside the pre-declared primary families.
- Not converting arrays to `cvxopt.matrix`; dense G for the SVM dual; leaving CVXOPT progress output on; demanding 1e-10 tolerances without the ladder.
- SMO stuck because the kernel is not precomputed or features are badly scaled; dividing by zero-variance pixels when standardising.
- Treating pᵀHp > 0 as evidence that an operator is symmetric (it is not); replacing a zero SMO curvature by an arbitrary floor instead of comparing the two endpoints.
- Quoting results, versions or counts from memory instead of from pasted output.

---

## 10. Parked work (not part of the course submission; do not implement unless asked)

- **E6 (P2)** patch-SVR reconstruction vs optimisation (7×7 zero-filled patch, `sklearn` SVR, compared with M2 at R = 4).
- **M11 (P2)** LS-SVM trained with CG, with a head-to-head table.
- **E11 (P2)** genuine raw k-space replication with fastMRI single-coil knee data, kept separate from Cheng classification.
- Rejected to protect scope: deep-learning reconstruction, ADMM, wavelets, more SVM kernels, three-class classification.

---

## 11. Reference seeds (students verify each entry against the publisher record before citing)

- Lustig, Donoho, Pauly, "Sparse MRI: The application of compressed sensing for rapid MR imaging", Magn. Reson. Med. 58(6), 2007.
- Donoho, "Compressed sensing", IEEE Trans. Inf. Theory 52(4), 2006.
- Rudin, Osher, Fatemi, "Nonlinear total variation based noise removal algorithms", Physica D 60, 1992.
- Vogel and Oman, "Iterative methods for total variation denoising", SIAM J. Sci. Comput. 17(1), 1996. (Check the DOI on the SIAM page; an earlier circulated DOI was wrong.)
- Dobson and Vogel, "Convergence of an iterative method for total variation denoising", SIAM J. Numer. Anal. 34(5), 1997.
- Kaipio and Somersalo, "Statistical inverse problems: Discretization, model reduction and inverse crimes", J. Comput. Appl. Math. 198(2), 2007.
- Shimron et al., "Implicit data crimes: Machine learning bias arising from misuse of public data", PNAS 119(13), 2022.
- Nadeau and Bengio, "Inference for the generalization error", Machine Learning 52(3), 2003.
- Wang, Bovik, Sheikh, Simoncelli, "Image quality assessment: from error visibility to structural similarity", IEEE Trans. Image Process. 13(4), 2004.
- Cheng et al., the two PLoS ONE papers cited on the figshare dataset page (2015 and 2016); take exact titles from that page.
- Hestenes and Stiefel (1952); Shewchuk, CG tutorial (1994); Cortes and Vapnik (1995); Platt (1998); Keerthi et al. and Fan, Chen, Lin on working-set selection for SMO; Chang and Lin, LIBSVM (2011); Jolliffe and Cadima (2016); Holm (1979). Course texts: Strang, *Linear Algebra and Learning from Data*; Boyd and Vandenberghe, *Convex Optimization*.
- Unverified entry carried over from an earlier plan: Morshuis, Hein, Baumgartner, arXiv:2508.18975 (reconstruction fidelity vs downstream segmentation). Confirm it exists and says what is claimed before citing.

---

## 12. Change log: v3.1 → v4.0

- **Assistant protocol (new):** assumes no code execution (students run and paste output); no invented versions, counts or DOIs; `pip freeze` for requirements; complete files only; Conventions Card re-pasted each milestone; per-milestone prompts (Section 8).
- **Tuning:** per-(method, R, SNR) λ and μ, wider M1 grid, an edge-extension rule, and a freeze rule; separate 16×16 λ grid for E3.
- **Noise:** generated as M ⊙ F(real image noise), so it is conjugate-symmetric and the nominal SNR is exact.
- **SMO:** replaced the random-second-index simplified SMO with deterministic maximal-violating-pair SMO and libsvm's stopping gap; added the dual-objective protocol for `SVC`.
- **CVXOPT:** tolerance ladder instead of demanding 1e-10 from the start; sparse G; documented ridge.
- **E4 leakage and inference:** PCA refit inside every inner fold; Protocol B defined precisely; one source of p-values (Nadeau–Bengio) and one of CIs (patient-clustered bootstrap) with a joint claim rule; continuous signed-margin outcome added to the association analysis; majority-class reference.
- **Data:** loader verification, image-size audit, `MIN_ROI_PIXELS`, seeded slice selection, adaptive cohort sizes, slice-orientation caveat.
- **Scope control:** G-BENCH timing gate after M3; explicit Floor and extended cut order.
- **Paper readiness:** E8 inverse-crime sensitivity (P1), figure and LaTeX-table scripts, `headline.csv`, manifest traceability, headline RQ3 identified.
- **Restored from v2:** E7 specification and the reading list (now flagged for verification).
- **Revision after an external review:** CG now separates the curvature check (positive definiteness) from an independent `check_symmetry` test; SMO handles zero curvature by comparing the two endpoints instead of using a floor, with a duplicate-row test; added an attrition table and an ROI-filter audit (slices and patients excluded per class, mask-area quantiles before and after); E5 no longer applies the ROI-size rule.

---

## Appendix S. For the students only (the assistant ignores this)

**Non-code items the course sheet requires (you write or produce these yourselves):** IEEE-format report (title page, abstract 200–250 words, introduction, literature review with at least 5–7 references, mathematical formulation, methodology, results, conclusion, references, appendix), contribution statement signed by all members, presentation slides, source code with README, dataset access instructions, Turnitin report (< 10% similarity, AI similarity < 20%). Late submissions lose 10% per day.

**Suggested split:** Student A: `mri_ops`, `cg`, `recon`, `qp_reference`, E1–E3. Student B: `data`, `pca`, `svm`, `stats`, E4–E5. Each reviews the other's code line by line and must be able to derive it; the viva can sample either of you on anything.

**Questions for your instructor:** (1) Is an MRI/TV application acceptable under Theme A (the sheet lists TV denoising as a Theme E example)? (2) Does a CVXOPT-based dual-QP SVM plus your own SMO count as "SVM from scratch"?

**Suggested 10-week schedule (compress using the Floor in 5.3 if your deadline is earlier):** W1 environment, data download, M1 start; W2 M1 done, M2; W3 M3, G-BENCH; W4 M4, M6 start; W5 M5 (tuning, E1–E3); W6 M6 done, folds and checks; W7 E4, E5; W8 M8, figures, P1 only if ahead; W9 report draft, slides, Turnitin pre-check; W10 revisions, clean-clone reproduction, submit at least 2 days early.

**If you later write a conference paper:** make RQ3 the headline, keep CG conditioning and the SVM-solver comparison for the course report, write the paper text fresh (not a trimmed copy of the report), and disclose AI-assistant use in the acknowledgments as IEEE requires.
