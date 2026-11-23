import numpy as np

import config
from src import cache


def test_key_changes_with_any_setting(monkeypatch):
    base = dict(slice=3, method="M1", R=4, snr=25.0, lam=1e-2, mu=None, seeds=[(4, 3)])
    k0 = cache.cache_key(base)
    for field, val in [("slice", 4), ("method", "M2"), ("R", 6), ("snr", float("inf")), ("lam", 2e-2), ("mu", 1e-3), ("seeds", [(4, 4)])]:
        assert cache.cache_key({**base, field: val}) != k0
    assert cache.cache_key(dict(reversed(list(base.items())))) == k0  # order independent
    monkeypatch.setattr(config, "CG_RTOL", 1e-9)  # config hash changes the key
    assert cache.cache_key(base) != k0


def test_roundtrip_and_single_compute(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return np.arange(6.0).reshape(2, 3), {"matvecs": 7, "snr": float("inf")}

    a1, m1 = cache.get_or_compute({"a": 1}, fn)
    a2, m2 = cache.get_or_compute({"a": 1}, fn)
    assert len(calls) == 1 and np.array_equal(a1, a2) and m1 == m2 and a2.dtype == np.float64
    cache.get_or_compute({"a": 2}, fn)
    assert len(calls) == 2
