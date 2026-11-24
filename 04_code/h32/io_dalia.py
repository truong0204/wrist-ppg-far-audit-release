"""PPG-DaLiA loader with a hard leakage barrier.

SX.pkl is a Python-2 pickle; a restricted unpickler admits only numpy array reconstruction.
Caches (02_data/derived/, git-ignored):
  * development subjects (DEV): SX.npz with bvp, acc, label, activity;
  * confirmatory subjects: SX.labels_acc.npz with acc, label, activity only — their PPG is not cached.
`load_subject(sid, need_ppg=True)` raises for a confirmatory subject unless the PPG gate in
05_experiments/prereg_v2.md has been opened (CONFIRMATORY_PPG_OPEN = True, set only in a commit after
the prereg v2 freeze).
"""
from __future__ import annotations
import importlib, pathlib, pickle
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "02_data" / "raw" / "dalia" / "extracted" / "PPG_FieldStudy"
DERIVED = ROOT / "02_data" / "derived"
DEV = ("S10", "S11", "S12")                       # prereg v1 P1 / v2 §1
CONFIRMATORY = tuple(f"S{i}" for i in range(1, 16) if f"S{i}" not in DEV)   # v2 naming; role changed in v2.1
# prereg v2.1 §1 (commit 9329ba2): all DaLiA subjects are training/development; PPG opened for training only.
CONFIRMATORY_PPG_OPEN = True

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        module = module.replace("numpy.core", "numpy._core")
        return getattr(importlib.import_module(module), name)


def _load_pkl(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return _SafeUnpickler(f, encoding="latin1").load()


def _build_cache(sid: str):
    d = _load_pkl(RAW / sid / f"{sid}.pkl")
    w = d["signal"]["wrist"]
    common = dict(acc=np.asarray(w["ACC"], float), label=np.asarray(d["label"], float).ravel(),
                  activity=np.asarray(d["activity"], float).ravel())
    np.savez(DERIVED / f"{sid}.npz", bvp=np.asarray(w["BVP"], float).ravel(), **common)


def load_subject(sid: str, need_ppg: bool = True) -> dict:
    """dict(bvp?, acc[M,3] (g), label[W] HR-ECG, activity[A], fs_*). bvp only for DEV (or open gate)."""
    if need_ppg and sid not in DEV and not CONFIRMATORY_PPG_OPEN:
        raise PermissionError(f"PPG of confirmatory subject {sid} is locked until prereg v2 opens it")
    DERIVED.mkdir(parents=True, exist_ok=True)
    cache = DERIVED / f"{sid}.npz"
    if not cache.exists():
        old = DERIVED / f"{sid}.labels_acc.npz"
        if old.exists() and not need_ppg:
            cache = old
        else:
            _build_cache(sid)
    z = np.load(cache)
    out = dict(sid=sid, acc=z["acc"], label=z["label"], activity=z["activity"], fs_bvp=64, fs_acc=32, fs_act=4)
    if need_ppg:
        out["bvp"] = z["bvp"]
    return out


def window_activity(activity: np.ndarray, n_windows: int, fs_act: int = 4, win_s: int = 8, shift_s: int = 2) -> np.ndarray:
    """Majority oracle activity ID per window (analysis only; never used by a method)."""
    out = np.zeros(n_windows, int)
    for i in range(n_windows):
        seg = activity[i * shift_s * fs_act:(i * shift_s + win_s) * fs_act].astype(int)
        out[i] = np.bincount(seg).argmax() if seg.size else -1
    return out
