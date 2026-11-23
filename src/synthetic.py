"""Synthetic Cheng-like .mat dataset for smoke tests ONLY. Results from it are not scientific results."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import savemat


def _phantom(n, rng, label, t, base_seed):
    """Brain-like slice with class-dependent tumor; t in [0,1] is the position along the patient's slices."""
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    head = (xx / 0.46) ** 2 + (yy / 0.40) ** 2 < 1
    brain = (xx / 0.40) ** 2 + (yy / 0.34) ** 2 < 1
    tex = gaussian_filter(np.random.default_rng(base_seed).standard_normal((n, n)), n / 14)
    img = 0.15 * head + 0.35 * brain + 0.60 * tex / tex.std() * brain
    ramp = np.random.default_rng(base_seed + 2).normal(0, 0.2, 2)
    img = img + brain * (ramp[0] * xx + ramp[1] * yy)
    ang = np.random.default_rng(base_seed + 1).uniform(0, 2 * np.pi)
    s = 1.0 - 0.35 * abs(t - 0.5)
    if label == 2:        # glioma: large, ring-enhancing, central
        cx, cy, r = 0.08 * np.cos(ang), 0.08 * np.sin(ang), 0.15 * s
        d = np.hypot(xx - cx, yy - cy)
        m = d < r
        img = img + 0.5 * (d < r) * (d > 0.6 * r) - 0.1 * (d < 0.6 * r) + 0.05 * rng.standard_normal((n, n)) * m
    elif label == 1:      # meningioma: compact, bright, near the skull
        cx, cy, r = 0.30 * np.cos(ang), 0.26 * np.sin(ang), 0.09 * s
        d = np.hypot(xx - cx, yy - cy)
        m = d < r
        img = img + 0.65 * m
    else:                 # pituitary (excluded by the binary filter)
        cx, cy, r = 0.0, 0.12, 0.05 * s
        m = np.hypot(xx - cx, yy - cy) < r
        img = img + 0.4 * m
    return img + 0.01 * rng.standard_normal((n, n)), m


def _write(path, pid, label, img, mask, kind):
    img16 = np.round(1000 * img).astype(np.int16)
    if kind == "h5":
        import h5py
        with h5py.File(path, "w") as f:
            g = f.create_group("cjdata")
            g.create_dataset("image", data=img16.T)                 # MATLAB v7.3 stores transposed
            g.create_dataset("tumorMask", data=mask.astype(np.uint8).T)
            g.create_dataset("label", data=np.array([[float(label)]]))
            g.create_dataset("PID", data=np.array([[ord(c)] for c in pid], dtype=np.uint16))
    else:
        savemat(path, {"cjdata": {"PID": pid, "label": float(label), "image": img16, "tumorMask": mask.astype(np.uint8)}})


def make_synthetic_dataset(out_dir, n_glioma=30, n_mening=26, n_pit=6, sizes=(128, 192), seed=0, inject=True):
    """Write .mat files (mixed v7.3 / v5); returns dict of injected special files for the tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plan = [(2, n_glioma), (1, n_mening), (3, n_pit)]
    k = 1
    info = dict(exact_dup=None, cross_class=None, mixed_pid=None, constant=None, tiny=None, files=[])
    made = []
    for label, cnt in plan:
        for j in range(cnt):
            pid = f"P{label}{j:03d}"
            n = sizes[(k + j) % len(sizes)]
            ns = int(rng.integers(3, 6))
            for s in range(ns):
                img, m = _phantom(n, rng, label, s / max(ns - 1, 1), 1000 * k + 10 * s)
                made.append((str(k), pid, label, img, m))
                k += 1
    if inject:
        gl = [r for r in made if r[2] == 2]
        a = gl[0]
        info["exact_dup"] = (a[0], str(k))
        made.append((str(k), a[1], 2, a[3].copy(), a[4].copy())); k += 1            # exact duplicate, same PID
        b = gl[5]
        info["cross_class"] = (b[0], str(k))
        made.append((str(k), "PX001", 1, b[3].copy(), b[4].copy())); k += 1         # same image labelled meningioma
        c = [r for r in made if r[2] == 1][3]
        info["mixed_pid"] = c[1]
        made.append((str(k), c[1], 2, [r for r in made if r[2] == 2][10][3] * 0.97, [r for r in made if r[2] == 2][10][4])); k += 1  # PID with two labels
        info["constant"] = str(k)
        made.append((str(k), "PC001", 2, np.full((128, 128), 0.3), np.zeros((128, 128), bool))); k += 1
        tiny = np.zeros((128, 128), bool); tiny[60:63, 60:63] = True
        info["tiny"] = str(k)
        made.append((str(k), "PT001", 1, _phantom(128, rng, 1, 0.5, 99991)[0], tiny)); k += 1
    for i, (fid, pid, label, img, m) in enumerate(made):
        _write(out_dir / f"{fid}.mat", pid, label, img, m, "h5" if i % 2 == 0 else "v5")
        info["files"].append(fid)
    return info
