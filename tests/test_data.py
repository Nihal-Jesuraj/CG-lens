import numpy as np
import pandas as pd
import pytest

import config
from src import data
from src.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    raw, proc, res = root / "raw", root / "proc", root / "res"
    info = make_synthetic_dataset(raw, n_glioma=24, n_mening=22, n_pit=4)
    res.mkdir()
    meta = data.prepare_raw(raw, proc)
    df, att, audit = data.filter_and_group(None, proc, res)
    return dict(raw=raw, proc=proc, res=res, info=info, meta=meta, df=df, att=att, audit=audit)


def test_loaders_agree_and_decode(tmp_path):
    from src.synthetic import _phantom, _write
    rng = np.random.default_rng(0)
    img, m = _phantom(64, rng, 2, 0.5, 5)
    _write(tmp_path / "1.mat", "PID042", 2, img, m, "h5")
    _write(tmp_path / "2.mat", "PID042", 2, img, m, "v5")
    a, b = data.load_mat(tmp_path / "1.mat"), data.load_mat(tmp_path / "2.mat")
    assert (a["backend"], b["backend"]) == ("h5py", "scipy")
    assert a["pid"] == b["pid"] == "PID042" and a["label"] == b["label"] == 2
    assert a["image"].shape == (64, 64)                       # transposition undone
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])
    assert a["raw_dtype"] == "int16"


def test_area_resize():
    rng = np.random.default_rng(1)
    a = rng.random((64, 64))
    assert np.abs(data.area_resize(a, (16, 16)) - a.reshape(16, 4, 16, 4).mean(axis=(1, 3))).max() < 1e-14
    b = rng.random((100, 90))
    r = data.area_resize(b, (30, 20))
    assert abs(r.mean() - b.mean()) < 1e-12 and r.dtype == np.float64
    assert np.allclose(data.area_resize(np.full((50, 50), 3.0), (7, 9)), 3.0)


def test_dhash_variants_and_popcount():
    rng = np.random.default_rng(2)
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(rng.random((128, 128)), 6)
    v = data.dhash_variants(img)
    for k in range(4):
        assert data.dhash(np.rot90(img, k)) == v[k]
    assert data.dhash(np.fliplr(img)) == v[4]
    pairs = data.near_dup_pairs(np.stack([v, data.dhash_variants(np.rot90(img)), data.dhash_variants(gaussian_filter(rng.random((128, 128)), 6))]))
    assert (0, 1) in pairs and (0, 2) not in pairs
    assert data.popcount64(np.array([0, 1, 2 ** 64 - 1], dtype=np.uint64)).tolist() == [0, 1, 64]


def test_dihedral_rmse_and_confirmation_threshold():
    rng = np.random.default_rng(2301)
    image = rng.uniform(0.2, 0.8, size=(128, 128))
    image[0, 0], image[0, 1] = 0.0, 1.0
    assert data.dihedral_rmse(image, image) == 0.0
    assert data.dihedral_rmse(image, np.rot90(image, 1)) == 0.0
    assert data.dihedral_rmse(image, np.fliplr(image)) == 0.0

    eligible = np.arange(2, image.size)
    below_idx = rng.choice(eligible, size=4095, replace=False)
    above_idx = rng.choice(eligible, size=4097, replace=False)
    below, above = image.copy(), image.copy()
    below.flat[below_idx] += 0.1
    above.flat[above_idx] += 0.1
    below_rmse = data.dihedral_rmse(image, below)
    above_rmse = data.dihedral_rmse(image, above)
    threshold = config.DHASH_CONFIRM_RMSE
    assert below_rmse < threshold and below_rmse <= threshold
    assert above_rmse > threshold and not above_rmse <= threshold


def test_patient_label_permutation_is_reproducible():
    from scripts.audit_dedup import patient_label_permutation

    frame = pd.DataFrame({"pid": ["p1", "p1", "p2", "p3", "p4", "p5"],
                          "label": [1, 1, 1, 1, 2, 2]})
    candidates = {"cross_label": [(0, 4, 2), (2, 5, 3)],
                  "same_label_same_PID": [], "same_label_different_PID": []}
    a = patient_label_permutation(candidates, frame, n_permutations=100, seed=17)
    b = patient_label_permutation(candidates, frame, n_permutations=100, seed=17)
    assert a == b
    assert a["observed_cross_label_candidates"] == 2
    assert a["n_permutations"] == 100 and a["analytic_expectation"] == 1.2


def test_e5_split_score_rewards_target_slice_patient_and_class_fractions():
    patient_labels = np.array([-1] * 5 + [1] * 5)
    y = np.repeat(patient_labels, 2)
    pids = np.repeat([f"p{i}" for i in range(10)], 2)
    balanced = np.array([0, 1, 10, 11])
    imbalanced = np.array([0, 1, 2, 3])
    assert data.e5_split_score(balanced, y, pids, 0.2) == 0.0
    assert data.e5_split_score(imbalanced, y, pids, 0.2) == 0.2


def test_union_find():
    uf = data.UnionFind(6)
    uf.union(0, 1); uf.union(2, 3); uf.union(1, 3)
    lab = uf.labels()
    assert lab[0] == lab[1] == lab[2] == lab[3] and lab[4] != lab[5] and lab[4] != lab[0]


def test_attrition_and_injected_cases(synth):
    att, df, info, audit = synth["att"], synth["df"], synth["info"], synth["audit"]
    tot = att.slices_total.tolist()
    assert all(a >= b for a, b in zip(tot, tot[1:]))
    assert att.stage.tolist()[:3] == ["raw files", "binary label filter", "label-consistent PID"]
    ids = set(df.id)
    assert info["mixed_pid"] in audit["pids_with_multiple_labels"] and info["mixed_pid"] not in set(df.pid)
    assert info["constant"] not in ids and audit["n_constant_images_removed"] == 1
    a, b = info["exact_dup"]
    assert a in ids and b not in ids                              # one copy kept
    c1, c2 = info["cross_class"]
    assert c1 not in ids and c2 not in ids                        # cross-class duplicates both dropped
    assert not df.roi_ok[df.id == info["tiny"]].any()
    roi = pd.read_csv(synth["res"] / "roi_filter_audit.csv")
    assert roi.slices_excluded.sum() >= 1 and set(roi.cls) == {"glioma", "meningioma"}
    assert (synth["res"] / "data_audit.json").exists()
    assert {"h": 128, "w": 128} in [{k: s[k] for k in ("h", "w")} for s in audit["image_size_histogram"]]
    assert len(audit["image_size_histogram"]) >= 2               # sizes are measured, not assumed


def test_intensity_convention(synth):
    df, img, msk = data.load_slices(synth["proc"])
    assert img.dtype == np.float64 and img.shape[1:] == (128, 128)
    assert np.allclose(img.min(axis=(1, 2)), 0) and np.allclose(img.max(axis=(1, 2)), 1)


def test_cohorts(synth, monkeypatch):
    for k, v in dict(RECON_VAL_PATIENTS=6, CLF_POOL_PATIENTS=30, RECON_EVAL_PATIENTS=12).items():
        monkeypatch.setattr(config, k, v)
    s = data.build_cohorts(synth["proc"], synth["res"])
    df = pd.read_csv(synth["proc"] / "slices.csv", dtype={"id": str, "pid": str})
    val, pool, ev = s["recon_val"], s["clf_pool"], s["recon_eval"]
    assert not set(val["patients"]) & set(pool["patients"])
    g = lambda ps: set(df[df.pid.isin(ps)].group)
    assert not g(val["patients"]) & g(pool["patients"])
    assert set(ev["patients"]) <= set(pool["patients"])
    for c in (val, pool, ev):
        sub = df[df["index"].isin(c["slice_index"])]
        assert sub.groupby("pid").size().max() <= config.MAX_SLICES_PER_PATIENT and sub.roi_ok.all()
        assert set(sub.pid) == set(c["patients"])
    assert val["n_patients"] == 6 and set(val["n_patients_by_label"]) == {"1", "2"} and min(val["n_patients_by_label"].values()) >= 1
    e5 = s["E5_pool"]
    tr, te = df.iloc[e5["train_index"]], df.iloc[e5["test_index"]]
    assert not set(tr.pid) & set(te.pid) and not set(tr.group) & set(te.group)
    assert len(tr) + len(te) == len(df)
    again = data.build_cohorts(synth["proc"], synth["res"])      # deterministic
    assert again["clf_pool"]["slice_index"] == pool["slice_index"]
    assert (synth["res"] / "splits.json").exists()


def test_extract_hires(synth):
    df, img, _ = data.load_slices(synth["proc"])
    ix = [int(df["index"].iloc[3])]
    hi = data.extract_hires(df, ix, 256, synth["raw"])[ix[0]]
    assert hi.shape == (256, 256)
    # area-average of the 256 version equals the 128 ground truth when native size >= 256 (checked for 192 -> loose)
    assert np.abs(data.area_resize(hi, (128, 128)) - img[ix[0]]).max() < 1e-9
