"""SHA-256 keyed .npy cache: a changed setting can never reuse a stale file."""
import hashlib
import json

import numpy as np

import config


def cache_key(key_dict):
    """SHA-256 of canonical JSON of key_dict plus the config hash."""
    payload = config.canonical_json({"key": key_dict, "config": config.config_hash()})
    return hashlib.sha256(payload.encode()).hexdigest()


def _paths(key):
    d = config.CACHE_DIR
    return d / f"{key}.npy", d / f"{key}.json"


def get_or_compute(key_dict, fn):
    """Return (array, meta). fn() -> (array, meta dict); computed once per key."""
    key = cache_key(key_dict)
    npy, js = _paths(key)
    if npy.exists() and js.exists():
        return np.load(npy), json.loads(js.read_text())
    arr, meta = fn()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, np.asarray(arr, dtype=np.float64))
    js.write_text(config.canonical_json(meta))
    return np.asarray(arr, dtype=np.float64), json.loads(config.canonical_json(meta))


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
