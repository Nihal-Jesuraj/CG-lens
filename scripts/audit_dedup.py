"""Audit dHash collisions against within-patient and cross-patient controls."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

import config
from src import data


def prepare():
    """Load the exact binary, label-consistent, exact-deduplicated audit cohort."""
    meta = pd.read_csv(config.PROC_DIR / "raw_meta.csv", dtype={"id": str, "pid": str})
    z = np.load(config.PROC_DIR / "raw128.npz")
    meta["raw_pos"] = np.arange(len(meta))
    df = meta[meta.label.isin([1, 2])].copy()
    nlab = df.groupby("pid").label.nunique()
    bad_pids = set(nlab[nlab > 1].index)
    df = df[~df.pid.isin(bad_pids)]
    df = df.drop_duplicates(subset=["sha", "label"], keep="first").reset_index(drop=True)
    images = z["image"]
    return df, images


def scan_pairs(df, images, max_distance, chunk=64):
    """Return distance counts and candidate pairs using the pipeline's exact dHash rule."""
    constant = (df.vmax128 - df.vmin128).to_numpy() <= 0
    eligible = np.flatnonzero(~constant)
    hashes = np.zeros((len(df), 8), dtype=np.uint64)
    positions = df.raw_pos.to_numpy()
    for i in eligible:
        hashes[i] = data.dhash_variants(data.minmax(images[positions[i]]))

    labels, pids = df.label.to_numpy(), df.pid.to_numpy()
    categories = ("cross_label", "same_label_same_PID", "same_label_different_PID")
    counts = np.zeros((len(categories), max_distance + 1), dtype=np.int64)
    candidates = {name: [] for name in categories}
    base = hashes[:, 0]
    for start in range(0, len(df), chunk):
        distances = data.popcount64(hashes[start:start + chunk, :, None] ^ base[None, None, :]).min(axis=1)
        for offset, row in enumerate(distances):
            i = start + offset
            if constant[i]:
                continue
            js = np.arange(i + 1, len(df))
            if not len(js):
                continue
            d = row[js]
            within = d <= max_distance
            js, d = js[within], d[within]
            for j, dist in zip(js, d):
                if labels[i] != labels[j]:
                    category = "cross_label"
                elif pids[i] == pids[j]:
                    category = "same_label_same_PID"
                else:
                    category = "same_label_different_PID"
                counts[categories.index(category), int(dist)] += 1
                if dist <= config.DHASH_MAX_HAMMING:
                    candidates[category].append((int(i), int(j), int(dist)))
    return counts, candidates, categories, int(constant.sum())


def max_ssim_dihedral(a, b):
    """Maximum SSIM over the eight alignments after independent min-max scaling."""
    a, b = data.minmax(a), data.minmax(b)
    best_ssim, best_rmse = -np.inf, np.nan
    for flip in (False, True):
        base = np.fliplr(a) if flip else a
        for rotation in range(4):
            aligned = np.rot90(base, rotation)
            score = structural_similarity(b, aligned, data_range=1.0)
            if score > best_ssim:
                best_ssim = float(score)
                best_rmse = float(np.sqrt(np.mean((b - aligned) ** 2)))
    return best_ssim, best_rmse


def write_samples(df, images, candidates, sample_count, seed):
    """Write a reproducible sample table and a contact sheet for visual review."""
    rng = np.random.default_rng(seed)
    sampled = []
    for category, pairs in candidates.items():
        if not pairs:
            continue
        chosen = rng.choice(len(pairs), size=min(sample_count, len(pairs)), replace=False)
        for index in sorted(chosen):
            i, j, distance = pairs[int(index)]
            a = images[int(df.iloc[i].raw_pos)]
            b = images[int(df.iloc[j].raw_pos)]
            best_ssim, best_rmse = max_ssim_dihedral(a, b)
            sampled.append(dict(category=category, id_a=df.iloc[i].id, pid_a=df.iloc[i].pid, label_a=int(df.iloc[i].label),
                                id_b=df.iloc[j].id, pid_b=df.iloc[j].pid, label_b=int(df.iloc[j].label),
                                hamming_distance=distance, max_ssim_dihedral=best_ssim, rmse_at_max_ssim=best_rmse,
                                raw_pos_a=int(df.iloc[i].raw_pos), raw_pos_b=int(df.iloc[j].raw_pos)))
    out = config.RESULTS_DIR
    sample_df = pd.DataFrame(sampled)
    sample_df.to_csv(out / "dedup_pair_samples.csv", index=False)

    scale, image_size, row_height = 2, 256, 286
    margin, title_height = 24, 44
    width = margin * 2 + image_size * 2 + 360
    height = margin + len(sampled) * row_height + title_height * len(candidates) + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    y = margin
    current = None
    for row in sampled:
        if row["category"] != current:
            current = row["category"]
            draw.text((margin, y), f"{current}: dHash distance <= {config.DHASH_MAX_HAMMING}", fill="black", font=font)
            y += title_height
        a, b = row["raw_pos_a"], row["raw_pos_b"]
        for x, pos, id_, pid, label in ((margin, a, row["id_a"], row["pid_a"], row["label_a"]),
                                        (margin + image_size + 12, b, row["id_b"], row["pid_b"], row["label_b"])):
            arr = np.clip(data.minmax(images[pos]) * 255, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((image_size, image_size), Image.Resampling.NEAREST).convert("RGB")
            sheet.paste(im, (x, y))
            draw.text((x, y + image_size + 4), f"id={id_} PID={pid} label={label}", fill="black", font=small)
        draw.text((margin + image_size * 2 + 40, y + 40),
                  f"distance={row['hamming_distance']}\nmax SSIM={row['max_ssim_dihedral']:.3f}\nRMSE={row['rmse_at_max_ssim']:.3f}",
                  fill="black", font=small, spacing=8)
        y += row_height
    sheet.crop((0, 0, width, y + margin)).save(out / "dedup_pair_samples.png")
    return sample_df


def all_cross_label_hamming_zero(df, images, candidates):
    """Exhaustively score cross-label candidate pairs at dHash distance zero."""
    rows = []
    for i, j, distance in candidates["cross_label"]:
        if distance != 0:
            continue
        a, b = images[int(df.iloc[i].raw_pos)], images[int(df.iloc[j].raw_pos)]
        score, rmse = max_ssim_dihedral(a, b)
        rows.append(dict(id_a=df.iloc[i].id, pid_a=df.iloc[i].pid, id_b=df.iloc[j].id, pid_b=df.iloc[j].pid,
                         label_a=int(df.iloc[i].label), label_b=int(df.iloc[j].label), hamming_distance=distance,
                         max_ssim_dihedral=score, rmse_at_max_ssim=rmse))
    return pd.DataFrame(rows)


def patient_label_permutation(candidates, df, n_permutations=10000, seed=config.GLOBAL_SEED):
    """Null distribution of cross-label dHash edges after shuffling labels across patients."""
    pids = sorted(df.pid.astype(str).unique())
    pid_index = {pid: i for i, pid in enumerate(pids)}
    labels_by_pid = (df.assign(pid=df.pid.astype(str)).drop_duplicates("pid").set_index("pid")
                     .loc[pids, "label"].to_numpy())
    pairs = [pair for group in candidates.values() for pair in group]
    left = np.fromiter((pid_index[str(df.iloc[i].pid)] for i, j, _ in pairs), dtype=np.int32, count=len(pairs))
    right = np.fromiter((pid_index[str(df.iloc[j].pid)] for i, j, _ in pairs), dtype=np.int32, count=len(pairs))
    observed = int(np.count_nonzero(labels_by_pid[left] != labels_by_pid[right]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.int64)
    for k in range(n_permutations):
        shuffled = rng.permutation(labels_by_pid)
        null[k] = np.count_nonzero(shuffled[left] != shuffled[right])
    values, counts = np.unique(labels_by_pid, return_counts=True)
    if len(values) == 2:
        n0, n1 = (int(x) for x in counts)
        between_pid_edges = int(np.count_nonzero(left != right))
        analytic_mean = between_pid_edges * (2 * n0 * n1) / (len(pids) * (len(pids) - 1))
    else:
        analytic_mean = float("nan")
    return dict(observed_cross_label_candidates=observed, candidate_pairs_total=len(pairs),
                n_patients=len(pids), patient_label_counts={str(int(v)): int(c) for v, c in zip(values, counts)},
                n_permutations=int(n_permutations), seed=int(seed),
                permutation_mean=float(null.mean()), permutation_sd=float(null.std(ddof=1)),
                permutation_p025=float(np.percentile(null, 2.5)), permutation_median=float(np.median(null)),
                permutation_p975=float(np.percentile(null, 97.5)), permutation_min=int(null.min()),
                permutation_max=int(null.max()), analytic_expectation=float(analytic_mean),
                lower_tail_probability=float((1 + np.count_nonzero(null <= observed)) / (n_permutations + 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-distance", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    parser.add_argument("--n-permutations", type=int, default=10000)
    args = parser.parse_args()
    config.ensure_dirs()
    df, images = prepare()
    counts, candidates, categories, n_constant = scan_pairs(df, images, args.max_distance)
    hist = pd.DataFrame(counts, index=categories, columns=[f"distance_{i}" for i in range(args.max_distance + 1)])
    hist.index.name = "pair_type"
    hist.to_csv(config.RESULTS_DIR / "dedup_distance_hist.csv")
    endpoints = set()
    for i, j, _ in candidates["cross_label"]:
        endpoints.update((i, j))
    summary = dict(n_binary_after_pid_and_exact_filters=len(df), n_constant_excluded=n_constant,
                   dHash_threshold=config.DHASH_MAX_HAMMING,
                   pairs_within_threshold={k: len(v) for k, v in candidates.items()},
                   cross_label_slices_removed_by_rule=len(endpoints),
                   mean_cross_label_degree=(2 * len(candidates["cross_label"]) / len(endpoints) if endpoints else 0),
                   max_distance_scanned=args.max_distance, sample_count=args.sample_count, seed=args.seed)
    samples = write_samples(df, images, candidates, args.sample_count, args.seed)
    cross_ssim = samples.loc[samples.category == "cross_label", "max_ssim_dihedral"]
    ssim_summary = dict(n_pairs=int(cross_ssim.notna().sum()),
                        min=float(cross_ssim.min()), max=float(cross_ssim.max()),
                        statistic="max SSIM across 8 dihedral alignments; independently min-max scaled images; skimage structural_similarity(data_range=1)")
    hamming_zero = all_cross_label_hamming_zero(df, images, candidates)
    hamming_zero.to_csv(config.RESULTS_DIR / "dedup_cross_label_hamming0_ssim.csv", index=False)
    zero_ssim = hamming_zero["max_ssim_dihedral"]
    zero_ssim_summary = dict(n_pairs=int(zero_ssim.notna().sum()), min=float(zero_ssim.min()), max=float(zero_ssim.max()),
                             statistic="exhaustive cross-label subset with dHash Hamming distance 0; max SSIM across 8 dihedral alignments; independently min-max scaled images")
    permutation = patient_label_permutation(candidates, df, args.n_permutations, args.seed)
    summary.update(sampled_cross_label_max_ssim=ssim_summary,
                   cross_label_hamming_zero_max_ssim=zero_ssim_summary,
                   patient_label_permutation=permutation)
    (config.RESULTS_DIR / "dedup_audit_summary.json").write_text(json.dumps(summary, indent=1))
    audit_path = config.RESULTS_DIR / "data_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["dhash_patient_label_permutation"] = permutation
    audit["sampled_cross_label_max_ssim"] = ssim_summary
    audit["cross_label_hamming_zero_max_ssim"] = zero_ssim_summary
    audit_path.write_text(json.dumps(audit, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"Observed cross-label candidates: {permutation['observed_cross_label_candidates']}")
    print(f"Patient-label permutation baseline ({permutation['n_permutations']} permutations, seed={permutation['seed']}): "
          f"mean={permutation['permutation_mean']:.1f}, SD={permutation['permutation_sd']:.1f}, "
          f"95% range=[{permutation['permutation_p025']:.1f}, {permutation['permutation_p975']:.1f}]")
    print(f"Sampled cross-label max-SSIM: n={ssim_summary['n_pairs']}, range=[{ssim_summary['min']:.4f}, {ssim_summary['max']:.4f}]")
    print(f"All cross-label dHash-0 max-SSIM: n={zero_ssim_summary['n_pairs']}, range=[{zero_ssim_summary['min']:.4f}, {zero_ssim_summary['max']:.4f}]")
    print("Wrote dedup_distance_hist.csv, dedup_pair_samples.csv, dedup_pair_samples.png, dedup_cross_label_hamming0_ssim.csv, dedup_audit_summary.json")


if __name__ == "__main__":
    main()
