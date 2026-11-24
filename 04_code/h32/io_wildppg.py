"""WildPPG loader with a hard gate (prereg v2.1 §1): PPG of confirmatory IDs is locked.

Per-ID polybox files: structs sternum/head/ankle/wrist with {fs, descr, v}; reference HR from the authors'
HF release (data_bpm_values, 8 s windows, 0 = removed). Index->ID mapping by recording length
(05_experiments/s1b/results/B_wildppg.json).
"""
from __future__ import annotations
import json, pathlib
import numpy as np, scipy.io as sio

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "02_data" / "raw" / "wildppg"
HF = ROOT / "02_data" / "raw" / "wildppg_hf" / "WildPPG.mat"
DEV = ("e61", "u7y", "ngh")
CONFIRMATORY = ("an0", "fex", "k2s", "kjd", "l38", "n31", "p5d", "p9p", "qm9", "ssx", "trh", "tz8", "w4p")
CONFIRMATORY_PPG_OPEN = True       # opened only by a commit after prereg v2.2


def hf_index(sid: str) -> int:
    m = json.load(open(ROOT / "05_experiments/s1b/results/B_wildppg.json"))["mapping"][sid]
    assert len(m) == 1
    return m[0]


def reference_hr(sid: str) -> np.ndarray:
    """Authors' cleaned HR, one value per 8 s window; 0 -> NaN. Label-only, allowed for every ID."""
    b = np.asarray(sio.loadmat(HF, variable_names=["data_bpm_values"])["data_bpm_values"][hf_index(sid), 0]).ravel().astype(float)
    b[b <= 0] = np.nan
    return b


def load_wrist(sid: str) -> dict:
    if sid not in DEV and not CONFIRMATORY_PPG_OPEN:
        raise PermissionError(f"WildPPG PPG of confirmatory subject {sid} is locked (prereg v2.1)")
    w = sio.loadmat(RAW / f"WildPPG_Part_{sid}.mat", variable_names=["wrist"], squeeze_me=True, struct_as_record=False)["wrist"]
    get = lambda k: (np.asarray(getattr(w, k).v, float), int(getattr(w, k).fs), str(getattr(w, k).descr))
    ppg, fs, d_ppg = get("ppg_g")
    ax, fsa, d_acc = get("acc_x"); ay, _, _ = get("acc_y"); az, _, _ = get("acc_z")
    return dict(sid=sid, ppg=ppg, fs_ppg=fs, acc=np.stack([ax, ay, az], 1), fs_acc=fsa,
                descr=dict(ppg=d_ppg, acc=d_acc), hr_ref=reference_hr(sid))
