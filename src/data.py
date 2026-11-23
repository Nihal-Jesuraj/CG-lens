"""Loader, audit, dHash + union-find groups, group-disjoint cohorts (plan Section 3).

CLI:  python -m src.data verify | audit | cohorts | manifest
"""
import functools
import json
import sys

import numpy as np
import pandas as pd

import config
from src import cache, mri_ops

QUANTILES = [0, 5, 25, 50, 75, 95, 100]


# ------------------------------------------------------------------ loading
def decode_pid(v):
    """PID from MATLAB v7.3 char codes, bytes, or str."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, str):
        return v
    arr = np.asarray(v).ravel()
    if arr.dtype.kind in "US":
        return "".join(str(s) for s in arr)
    return "".join(chr(int(c)) for c in arr)


def _field(obj, name):
    return getattr(obj, name) if hasattr(obj, name) else obj[name]


def load_mat(path):
    """Return dict(pid, label, image, mask, backend). h5py first (v7.3, transposed), scipy fallback."""
    import h5py
    path = str(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            g = f["cjdata"] if "cjdata" in f else f  # struct name 'cjdata' is an assumption; audit prints facts
            raw = np.array(g["image"])
            raw_dtype = str(raw.dtype)
            image = raw.T.astype(np.float64)                     # HDF5 returns MATLAB arrays transposed
            mask = np.array(g["tumorMask"]).T.astype(np.float64)
            label = int(np.array(g["label"]).ravel()[0])
            pid = decode_pid(np.array(g["PID"]))
        return dict(pid=pid, label=label, image=image, mask=mask, backend="h5py", raw_dtype=raw_dtype)
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m["cjdata"] if "cjdata" in m else m
    return dict(pid=decode_pid(_field(s, "PID")), label=int(np.asarray(_field(s, "label")).ravel()[0]),
                image=np.asarray(_field(s, "image"), dtype=np.float64), mask=np.asarray(_field(s, "tumorMask"), dtype=np.float64),
                backend="scipy", raw_dtype=str(np.asarray(_field(s, "image")).dtype))


def list_files(raw_dir):
    fs = list(raw_dir.glob("*.mat"))
    return sorted(fs, key=lambda p: (0, int(p.stem), "") if p.stem.isdigit() else (1, 0, p.stem))


def describe_files(raw_dir, k=5):
    """Print/return shapes, dtypes and decoded PID of the first k files (loader verification)."""
    facts = []
    for f in list_files(raw_dir)[:k]:
        d = load_mat(f)
        facts.append(dict(file=f.name, backend=d["backend"], pid=d["pid"], label=d["label"], image_shape=list(d["image"].shape),
                          mask_shape=list(d["mask"].shape), image_min=float(d["image"].min()), image_max=float(d["image"].max()),
                          mask_unique=[float(u) for u in np.unique(d["mask"])[:5]]))
        print(facts[-1])
    return facts


# ------------------------------------------------------------------ resizing, hashing
@functools.lru_cache(maxsize=None)
def _area_matrix(n_in, n_out):
    edges = np.linspace(0, n_in, n_out + 1)
    W = np.zeros((n_out, n_in))
    for o in range(n_out):
        lo, hi = edges[o], edges[o + 1]
        for i in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            W[o, i] = max(0.0, min(hi, i + 1) - max(lo, i))
    return W / (edges[1] - edges[0])


def area_resize(a, out_shape):
    """Exact float64 area-average (box) resampling to out_shape = (h, w)."""
    a = np.asarray(a, dtype=np.float64)
    return _area_matrix(a.shape[0], out_shape[0]) @ a @ _area_matrix(a.shape[1], out_shape[1]).T


def minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo)


def dhash(img):
    """64-bit difference hash: 8x9 area average, horizontal neighbour comparison."""
    s = area_resize(img, (8, 9))
    bits = (s[:, :-1] > s[:, 1:]).ravel()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def dhash_variants(img):
    """Hashes of the 8 dihedral variants (4 rotations x flip)."""
    out = []
    for flip in (False, True):
        base = np.fliplr(img) if flip else img
        for k in range(4):
            out.append(dhash(np.rot90(base, k)))
    return np.array(out, dtype=np.uint64)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def near_dup_pairs(variants, maxd=None, chunk=64):
    """Index pairs (i<j) whose hashes are within maxd over all dihedral variants."""
    maxd = config.DHASH_MAX_HAMMING if maxd is None else maxd
    n = len(variants)
    base = variants[:, 0]
    pairs = []
    for s in range(0, n, chunk):
        v = variants[s:s + chunk]                                  # (c, 8)
        d = popcount64(v[:, :, None] ^ base[None, None, :]).min(axis=1)  # (c, n)
        ii, jj = np.nonzero(d <= maxd)
        for a, b in zip(ii + s, jj):
            if a < b:
                pairs.append((int(a), int(b)))
    return pairs


def dihedral_rmse(a, b):
    """Minimum RMSE after min-max scaling and the same eight dHash transforms."""
    a, b = minmax(a), minmax(b)
    best = np.inf
    for flip in (False, True):
        base = np.fliplr(a) if flip else a
        for k in range(4):
            diff = np.rot90(base, k) - b
            best = min(best, float(np.mean(diff * diff)))
    return float(np.sqrt(best))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self):
        roots = [self.find(i) for i in range(len(self.p))]
        _, lab = np.unique(roots, return_inverse=True)
        return lab


# ------------------------------------------------------------------ prepare, audit, attrition
def _counts(df, stage):
    row = dict(stage=stage, slices_total=len(df), patients_total=df.pid.nunique())
    for name, lab in (("glioma", 2), ("meningioma", 1)):
        s = df[df.label == lab]
        row[f"slices_{name}"], row[f"patients_{name}"] = len(s), s.pid.nunique()
    o = df[~df.label.isin([1, 2])]
    row["slices_other"], row["patients_other"] = len(o), o.pid.nunique()
    return row


def prepare_raw(raw_dir=None, proc_dir=None):
    """Load every .mat once: exact hash + 128x128 area-averaged image and binarised mask."""
    raw_dir, proc_dir = raw_dir or config.RAW_DIR, proc_dir or config.PROC_DIR
    proc_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"no .mat files in {raw_dir}")
    rows, imgs, masks = [], [], []
    import hashlib
    for f in files:
        d = load_mat(f)
        im, mk = d["image"], d["mask"]
        h = hashlib.sha256(np.ascontiguousarray(im).tobytes() + str(im.shape).encode()).hexdigest()
        i128 = area_resize(im, (config.N, config.N))
        m128 = area_resize(mk, (config.N, config.N)) > 0.5
        rows.append(dict(id=f.stem, pid=d["pid"], label=d["label"], h=im.shape[0], w=im.shape[1], dtype=d["raw_dtype"], backend=d["backend"],
                         vmin=float(im.min()), vmax=float(im.max()), sha=h, mask_area_orig=int(np.count_nonzero(mk)),
                         img_mismatch=int(im.shape != mk.shape), vmin128=float(i128.min()), vmax128=float(i128.max())))
        imgs.append(i128)
        masks.append(m128)
    meta = pd.DataFrame(rows)
    np.savez(proc_dir / "raw128.npz", image=np.stack(imgs), mask=np.stack(masks))
    meta.to_csv(proc_dir / "raw_meta.csv", index=False)
    return meta


def filter_and_group(meta=None, proc_dir=None, results_dir=None):
    """Apply the attrition stages in plan order; write audit, attrition, ROI audit, slices.npz."""
    proc_dir, results_dir = proc_dir or config.PROC_DIR, results_dir or config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(proc_dir / "raw_meta.csv", dtype={"id": str, "pid": str}) if meta is None else meta
    z = np.load(proc_dir / "raw128.npz")
    image, mask = z["image"], z["mask"]
    meta = meta.copy()
    meta["raw_pos"] = np.arange(len(meta))
    att = [_counts(meta, "raw files")]

    df = meta[meta.label.isin([1, 2])]
    att.append(_counts(df, "binary label filter"))
    nlab = df.groupby("pid").label.nunique()
    bad_pids = sorted(nlab[nlab > 1].index)
    df = df[~df.pid.isin(bad_pids)]
    att.append(_counts(df, "label-consistent PID"))
    df = df.drop_duplicates(subset=["sha", "label"], keep="first")   # keep one per (image, label)
    att.append(_counts(df, "exact-duplicate removal"))

    const = (df.vmax128 - df.vmin128).to_numpy() <= 0
    pos = df.raw_pos.to_numpy()
    hv = np.zeros((len(df), 8), dtype=np.uint64)
    for k in np.flatnonzero(~const):
        hv[k] = dhash_variants(minmax(image[pos[k]]))
    nz = np.flatnonzero(~const)
    candidates = [(int(nz[a]), int(nz[b])) for a, b in near_dup_pairs(hv[nz])] if len(nz) else []
    labels = df.label.to_numpy()
    pids = df.pid.astype(str).to_numpy()
    category_names = ("same_label_same_PID", "same_label_different_PID", "cross_label")
    rmse_by_class = {name: [] for name in category_names}
    candidate_rmse = []
    pairs = []
    for a, b in candidates:
        rmse = dihedral_rmse(image[pos[a]], image[pos[b]])
        category = ("cross_label" if labels[a] != labels[b] else
                    "same_label_same_PID" if pids[a] == pids[b] else
                    "same_label_different_PID")
        rmse_by_class[category].append(rmse)
        candidate_rmse.append((a, b, rmse, category))
        if rmse <= config.DHASH_CONFIRM_RMSE:
            pairs.append((a, b))

    rmse_summary = {
        name: dict(n_pairs=len(values),
                   rmse_min=float(np.min(values)) if values else None,
                   rmse_p1=float(np.percentile(values, 1)) if values else None,
                   n_edges_at_threshold=int(np.sum(np.asarray(values) <= config.DHASH_CONFIRM_RMSE)))
        for name, values in rmse_by_class.items()
    }
    same_pid_min = rmse_summary["same_label_same_PID"]["rmse_min"]
    cross_label_min = rmse_summary["cross_label"]["rmse_min"]
    if same_pid_min is not None and cross_label_min is not None and same_pid_min < cross_label_min:
        safe_max = float(np.nextafter(cross_label_min, -np.inf))
        safe_cross_edges = int(sum(rmse <= safe_max for _, _, rmse, category in candidate_rmse
                                   if category == "cross_label"))
    else:
        safe_max, safe_cross_edges = None, None

    # Quantify the impact of applying PLAN_v4's dHash-only rule before the
    # conservative RMSE confirmation: drop cross-label endpoints, then group
    # surviving slices by PID and all same-label candidate edges.
    cross_candidates = [(a, b) for a, b in candidates if labels[a] != labels[b]]
    plan_dhash_cross_endpoints = {i for a, b in cross_candidates for i in (a, b)}
    plan_survivors = [i for i in range(len(df)) if i not in plan_dhash_cross_endpoints]
    plan_remap = {old: new for new, old in enumerate(plan_survivors)}
    plan_uf = UnionFind(len(plan_survivors))
    first_by_pid = {}
    for i in plan_survivors:
        plan_uf.union(plan_remap[i], plan_remap[first_by_pid.setdefault(pids[i], i)])
    for a, b in candidates:
        if labels[a] == labels[b] and a in plan_remap and b in plan_remap:
            plan_uf.union(plan_remap[a], plan_remap[b])
    plan_groups = {}
    for i in plan_survivors:
        plan_groups.setdefault(plan_uf.find(plan_remap[i]), []).append(i)
    largest_plan_group = max(plan_groups.values(), key=len, default=[])
    cross = {p for pr in pairs for p in pr if labels[pr[0]] != labels[pr[1]]}
    cross_pairs = sum(labels[a] != labels[b] for a, b in pairs)
    keep = np.array([k not in cross for k in range(len(df))])
    df2, const2 = df[keep].copy(), const[keep]
    att.append(_counts(df2, "cross-class near-duplicate removal"))
    df3 = df2[~const2].copy()
    att.append(_counts(df3, "constant-image removal"))
    n_const = int(const2.sum())

    # groups on the surviving set: union over PID and over same-class near-duplicate edges
    old2new = {old: new for new, old in enumerate(np.flatnonzero(keep)[~const2])}
    uf = UnionFind(len(df3))
    first = {}
    for k, p in enumerate(df3.pid):
        uf.union(k, first.setdefault(p, k))
    n_edges = 0
    for a, b in pairs:
        if a in old2new and b in old2new:
            uf.union(old2new[a], old2new[b])
            n_edges += 1
    df3["group"] = uf.labels()
    df3 = df3.reset_index(drop=True)
    df3["index"] = np.arange(len(df3))
    pos3 = df3.raw_pos.to_numpy()
    img = np.stack([minmax(image[p]) for p in pos3])           # ground truth min-max scaled once
    msk = mask[pos3].astype(np.uint8)
    df3["mask_area"] = msk.reshape(len(df3), -1).sum(axis=1)
    df3["roi_ok"] = df3.mask_area >= config.MIN_ROI_PIXELS
    roi = df3[df3.roi_ok]
    att.append(_counts(roi, "ROI-eligible"))
    df3["y"] = df3.label.map(config.BINARY_LABELS)
    np.savez(proc_dir / "slices.npz", image=img, mask=msk)
    df3.to_csv(proc_dir / "slices.csv", index=False)

    # cohort-selection row is appended by build_cohorts
    pd.DataFrame(att).to_csv(results_dir / "attrition.csv", index=False)

    rrows = []
    for name, lab in (("glioma", 2), ("meningioma", 1)):
        a, b = df3[df3.label == lab], df3[(df3.label == lab) & df3.roi_ok]
        excl = a[~a.roi_ok]
        lost = set(a.pid) - set(b.pid)
        row = dict(cls=name, slices_before=len(a), slices_after=len(b), slices_excluded=len(excl),
                   patients_affected=excl.pid.nunique(), patients_lost_all=len(lost))
        for q in QUANTILES:
            row[f"area_q{q}_before"] = float(np.percentile(a.mask_area, q)) if len(a) else np.nan
            row[f"area_q{q}_after"] = float(np.percentile(b.mask_area, q)) if len(b) else np.nan
        rrows.append(row)
    pd.DataFrame(rrows).to_csv(results_dir / "roi_filter_audit.csv", index=False)

    sizes = meta.groupby(["h", "w"]).size().reset_index(name="n")
    audit = dict(
        n_files=len(meta), counts_by_label={str(k): int(v) for k, v in meta.label.value_counts().items()},
        patients_by_label={str(k): int(v) for k, v in meta.groupby("label").pid.nunique().items()},
        pids_with_multiple_labels=bad_pids, image_size_histogram=[dict(h=int(r.h), w=int(r.w), n=int(r.n)) for r in sizes.itertuples()],
        image_mask_shape_mismatch=int(meta.img_mismatch.sum()), dtypes=sorted(meta.dtype.unique().tolist()),
        raw_intensity_min=float(meta.vmin.min()), raw_intensity_max=float(meta.vmax.max()),
        empty_masks_original=int((meta.mask_area_orig == 0).sum()), empty_masks_128=int((df3.mask_area == 0).sum()),
        mask_area_128_quantiles={str(q): float(np.percentile(df3.mask_area, q)) for q in QUANTILES},
        backends=sorted(meta.backend.unique().tolist()), n_constant_images_removed=n_const,
        n_dhash_candidate_pairs=len(candidates),
        n_dhash_candidate_cross_class_pairs=int(sum(labels[a] != labels[b] for a, b in candidates)),
        dhash_confirm_rmse=config.DHASH_CONFIRM_RMSE,
        dhash_candidate_rmse_by_class=rmse_summary,
        dhash_safe_threshold_interval=dict(
            lower_exclusive_same_PID_min=same_pid_min,
            upper_exclusive_cross_label_min=cross_label_min,
            max_safe_threshold=safe_max,
            n_cross_label_edges_at_max_safe_threshold=safe_cross_edges,
        ),
        plan_dhash_only_impact=dict(
            cross_label_endpoints_removed=len(plan_dhash_cross_endpoints),
            slices_remaining=len(plan_survivors),
            groups_after_same_label_merging=len(plan_groups),
            largest_group_slices=len(largest_plan_group),
            largest_group_patients=int(df.iloc[largest_plan_group].pid.nunique()) if largest_plan_group else 0,
        ),
        n_confirmed_near_dup_pairs=len(pairs),
        n_cross_class_near_dup_slices_removed=len(cross), n_cross_class_pairs=int(cross_pairs), n_same_class_edges=n_edges,
        n_groups=int(df3.group.nunique()), image_size_assumption="none (histogram above is measured)",
    )
    (results_dir / "data_audit.json").write_text(json.dumps(audit, indent=1))
    return df3, pd.DataFrame(att), audit


# ------------------------------------------------------------------ cohorts
def _targets(counts, total):
    """Class-proportional integer targets summing to total (>=1 per non-empty class)."""
    s = sum(counts.values())
    t = {c: (max(1, int(round(total * n / s))) if n else 0) for c, n in counts.items()}
    while sum(t.values()) > total:
        t[max(t, key=t.get)] -= 1
    while sum(t.values()) < total and any(t[c] < counts[c] for c in t):
        t[max((c for c in t if t[c] < counts[c]), key=lambda c: counts[c] - t[c])] += 1
    return t


def _pick_groups(groups, label_of, target, rng):
    """groups: dict gid -> list of patients; take whole groups per class up to target patients."""
    chosen, cnt = [], {c: 0 for c in target}
    for c in target:
        gids = [g for g in groups if label_of[g] == c]
        for k in rng.permutation(len(gids)):
            g = gids[k]
            if cnt[c] + len(groups[g]) <= target[c]:
                chosen.append(g)
                cnt[c] += len(groups[g])
    return chosen


def e5_split_score(test, y, pids, test_frac):
    """Score an E5 test fold by slice, patient, and class-specific slice fractions."""
    test = np.asarray(test, dtype=int)
    y, pids = np.asarray(y), np.asarray(pids)
    test_y = y[test]
    test_classes = np.array([np.sum(test_y == -1), np.sum(test_y == 1)], dtype=float)
    class_totals = np.array([np.sum(y == -1), np.sum(y == 1)], dtype=float)
    slice_error = abs(len(test) / len(y) - test_frac)
    patient_error = abs(len(np.unique(pids[test])) / len(np.unique(pids)) - test_frac)
    class_error = float(np.mean(np.abs(test_classes / class_totals - test_frac)))
    return float(slice_error + patient_error + class_error)


def build_cohorts(proc_dir=None, results_dir=None):
    """Seeded, group-disjoint, class-stratified cohorts -> results/splits.json."""
    from sklearn.model_selection import StratifiedGroupKFold
    proc_dir, results_dir = proc_dir or config.PROC_DIR, results_dir or config.RESULTS_DIR
    df = pd.read_csv(proc_dir / "slices.csv", dtype={"id": str, "pid": str})
    pids_sorted = sorted(df.pid.unique())
    pidx = {p: i for i, p in enumerate(pids_sorted)}
    el = df[df.roi_ok]
    chosen = {}
    for p, g in el.groupby("pid"):
        rng = mri_ops.seed_rng(7001, pidx[p])
        idx = g["index"].to_numpy()
        take = idx if len(idx) <= config.MAX_SLICES_PER_PATIENT else rng.choice(idx, config.MAX_SLICES_PER_PATIENT, replace=False)
        chosen[p] = sorted(int(i) for i in take)
    pt = el.groupby("pid").agg(label=("label", "first"), group=("group", "first")).reset_index()
    if pt.groupby("pid").group.nunique().max() > 1:  # pragma: no cover  (union-find guarantees one group per PID)
        raise RuntimeError("PID spans several groups")
    groups = {g: list(s.pid) for g, s in pt.groupby("group")}
    glabel = {g: int(pt[pt.group == g].label.iloc[0]) for g in groups}

    def counts_of(gs):
        out = {1: 0, 2: 0}
        for g in gs:
            out[glabel[g]] += len(groups[g])
        return out

    notes = []
    all_g = list(groups)
    n_total = sum(len(v) for v in groups.values())
    t_val = _targets(counts_of(all_g), min(config.RECON_VAL_PATIENTS, n_total))
    val_g = _pick_groups(groups, glabel, t_val, mri_ops.seed_rng(7002))
    rest = [g for g in all_g if g not in set(val_g)]
    n_rest = sum(len(groups[g]) for g in rest)
    n_pool = min(config.CLF_POOL_PATIENTS, n_rest)
    if n_pool < config.CLF_POOL_PATIENTS:
        notes.append(f"clf_pool shrunk to {n_pool} patients (eligible patients={n_total}; recon_val kept at {config.RECON_VAL_PATIENTS})")
    if n_pool == n_rest:
        pool_g = rest
    else:
        pool_g = _pick_groups({g: groups[g] for g in rest}, glabel, _targets(counts_of(rest), n_pool), mri_ops.seed_rng(7003))
    pool_p = sorted(p for g in pool_g for p in groups[g])
    plabel = dict(zip(pt.pid, pt.label))
    n_eval = min(config.RECON_EVAL_PATIENTS, len(pool_p))
    cnt = {c: sum(plabel[p] == c for p in pool_p) for c in (1, 2)}
    te = _targets(cnt, n_eval)
    rng = mri_ops.seed_rng(7004)
    eval_p = []
    for c in (1, 2):
        cand = [p for p in pool_p if plabel[p] == c]
        eval_p += [cand[k] for k in rng.permutation(len(cand))[:te[c]]]
    val_p = sorted(p for g in val_g for p in groups[g])

    def pack(pids):
        sl = [i for p in pids for i in chosen[p]]
        return dict(patients=sorted(pids), slice_index=sorted(sl), slice_id=[df.loc[i, "id"] for i in sorted(sl)],
                    n_patients=len(pids), n_slices=len(sl),
                    n_patients_by_label={str(c): sum(plabel[p] == c for p in pids) for c in (1, 2)})

    # E5 pool: all surviving slices (no ROI rule), one group-disjoint 80/20 split
    y = df.y.to_numpy()
    n_patients_total = df.pid.nunique()
    n_splits = int(round(1 / config.E5_TEST_FRAC))
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.GLOBAL_SEED)
    all_splits = list(sgk.split(np.zeros(len(df)), y, df.group.to_numpy()))
    # A single SGKF fold is not guaranteed to contain 20% of slices when
    # patient/duplicate groups vary in size. Select the fold closest to the
    # declared slice, patient, and class proportions, with fold order as tie-break.
    fold_id, (tr, te_) = min(enumerate(all_splits),
                             key=lambda item: (e5_split_score(item[1][1], y, df.pid.to_numpy(), config.E5_TEST_FRAC), item[0]))
    e5 = dict(train_index=[int(i) for i in tr], test_index=[int(i) for i in te_], n_slices=len(df), n_patients=int(df.pid.nunique()),
              train_patients=int(df.pid.iloc[tr].nunique()), test_patients=int(df.pid.iloc[te_].nunique()),
              fold_id=int(fold_id), n_splits=n_splits, test_fraction_slices=float(len(te_) / len(df)),
              test_fraction_patients=float(df.pid.iloc[te_].nunique() / n_patients_total),
              split_score=float(e5_split_score(te_, y, df.pid.to_numpy(), config.E5_TEST_FRAC)))
    assert not (set(df.pid.iloc[tr]) & set(df.pid.iloc[te_])) and not (set(df.group.iloc[tr]) & set(df.group.iloc[te_]))

    splits = dict(recon_val=pack(val_p), clf_pool=pack(pool_p), recon_eval=pack(eval_p), E5_pool=e5, notes=notes,
                  config_hash=config.config_hash(), seed=config.GLOBAL_SEED)
    gv, gp = {int(pt[pt.pid == p].group.iloc[0]) for p in val_p}, {int(pt[pt.pid == p].group.iloc[0]) for p in pool_p}
    assert not (set(val_p) & set(pool_p)) and not (gv & gp), "recon_val / clf_pool overlap"
    assert set(eval_p) <= set(pool_p)
    (results_dir / "splits.json").write_text(json.dumps(splits, indent=1))

    att = pd.read_csv(results_dir / "attrition.csv")
    sel = [dict(stage="cohort selection (recon_val + clf_pool)", slices_total=splits["recon_val"]["n_slices"] + splits["clf_pool"]["n_slices"],
                patients_total=len(val_p) + len(pool_p))]
    for name, lab in (("glioma", 2), ("meningioma", 1)):
        ps = [p for p in val_p + pool_p if plabel[p] == lab]
        sel[0][f"patients_{name}"] = len(ps)
        sel[0][f"slices_{name}"] = sum(len(chosen[p]) for p in ps)
    att = pd.concat([att[att.stage != sel[0]["stage"]], pd.DataFrame(sel)], ignore_index=True)
    att.to_csv(results_dir / "attrition.csv", index=False)
    return splits


def load_slices(proc_dir=None):
    proc_dir = proc_dir or config.PROC_DIR
    df = pd.read_csv(proc_dir / "slices.csv", dtype={"id": str, "pid": str})
    z = np.load(proc_dir / "slices.npz")
    return df, z["image"], z["mask"].astype(np.float64)


def load_splits(results_dir=None):
    return json.loads(((results_dir or config.RESULTS_DIR) / "splits.json").read_text())


def extract_hires(df, indices, size=256, raw_dir=None):
    """Re-read raw files for E8: area-averaged size x size image mapped by the 128-level min/max affine."""
    raw_dir = raw_dir or config.RAW_DIR
    out = {}
    for i in indices:
        r = df.loc[df["index"] == i].iloc[0]
        d = load_mat(raw_dir / f"{r.id}.mat")
        x = area_resize(d["image"], (size, size))
        out[int(i)] = (x - r.vmin128) / (r.vmax128 - r.vmin128)
    return out


def write_manifest(raw_dir=None, out=None):
    raw_dir = raw_dir or config.RAW_DIR
    out = out or (config.DATA_DIR / "MANIFEST.sha256")
    lines = [f"{cache.file_sha256(f)}  {f.name}" for f in list_files(raw_dir)]
    out.write_text("\n".join(lines) + "\n")
    return len(lines)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "audit"
    config.ensure_dirs()
    if cmd == "verify":
        facts = describe_files(config.RAW_DIR)
        (config.RESULTS_DIR / "loader_facts.json").write_text(json.dumps(facts, indent=1))
    elif cmd == "audit":
        prepare_raw()
        df, att, audit = filter_and_group()
        print(att.to_string(index=False))
        print(pd.read_csv(config.RESULTS_DIR / "roi_filter_audit.csv").T.to_string())
        print(json.dumps({k: audit[k] for k in ("counts_by_label", "patients_by_label", "image_size_histogram", "pids_with_multiple_labels")}, indent=1))
        print("\nSTOP: review these counts before running `python -m src.data cohorts`.")
    elif cmd == "cohorts":
        s = build_cohorts()
        print({k: (v["n_patients"], v["n_slices"]) for k, v in s.items() if isinstance(v, dict) and "n_patients" in v}, s["notes"])
    elif cmd == "manifest":
        print(write_manifest(), "files hashed")
    else:
        raise SystemExit("usage: python -m src.data verify|audit|cohorts|manifest")


if __name__ == "__main__":
    main(sys.argv)
