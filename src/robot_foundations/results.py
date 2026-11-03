"""Writing results and checking quoted numbers (examples/)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def clean(o):
    """Make numpy values JSON-safe (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def lookup(res, key):
    """res['a']['b'] for key 'a.b' (list items: 'a.0')."""
    for part in key.split("."):
        res = res[int(part) if isinstance(res, list) else part]
    return res


def check(res, expected):
    """Compare res with {key: (value, tol)}; list of misses."""
    bad = []
    for key, (want, tol) in expected.items():
        got = lookup(res, key)
        ok = got == want if tol is None else (
            got is not None and abs(got - want) <= tol)
        if not ok:
            bad.append(f"{key}: got {got!r}, want {want!r}"
                       f" +- {tol}")
    return bad


def write_arrays(out: Path, name, arrays, desc, meta):
    """name.npz plus the name.json sidecar for the visualizer.

    desc: {array: (units, what)}; meta: frame, time base ...
    """
    missing = set(arrays) ^ set(desc)
    if missing:
        raise KeyError(f"sidecar/arrays mismatch: {missing}")
    np.savez_compressed(out / f"{name}.npz", **arrays)
    side = dict(meta)
    side["arrays"] = {k: {"shape": list(np.shape(v)),
                          "dtype": str(np.asarray(v).dtype),
                          "units": desc[k][0],
                          "what": desc[k][1]}
                      for k, v in arrays.items()}
    (out / f"{name}.json").write_text(
        json.dumps(clean(side), indent=1))
