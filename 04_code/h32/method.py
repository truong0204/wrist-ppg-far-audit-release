"""Main method (prereg v2.2 DRAFT §2, 03_theory/R1.md) and threshold baselines on one window stream.

Windows: 8 s, stride 2 s; window i has centre time 2 i + 4 s.
- state s_i from ACC (v2 §3): 0 rest (< 0.02 g), 1 light (< 0.10 g), 2 vigorous; std of |acc| per window.
- monitored iff SQI_i >= theta.
- state-time blocks: per state, a clock counts monitored windows of that state; block j = windows with
  clock index in [j LW, (j + 1) LW), LW = 300 windows = 10 min of that state's monitored time.
- D_i = min(h_i, h_{i-1}, h_{i-2}) if the three windows are real-time consecutive, monitored, same state and
  same state-time block; else -inf.
- alarm starts at i iff D_i > thr_i, block (s_i, blk_i) is a test block, and no alarm has started in that
  block yet (refractory = one state-time block, on the state clock).
All thresholds share this machinery, so methods differ only in thr_i.
"""
from __future__ import annotations
import numpy as np

LW = 300
STATE_EDGES = (0.02, 0.10)


def states(acc32: np.ndarray, n: int) -> np.ndarray:
    s = np.array([np.std(np.linalg.norm(acc32[i * 64:i * 64 + 256], axis=1)) for i in range(n)])
    return np.digitize(s, STATE_EDGES)


def blocks(mon: np.ndarray, st: np.ndarray, lw: int = LW) -> np.ndarray:
    blk = np.full(len(mon), -1)
    clock = np.zeros(3, int)
    for i in np.flatnonzero(mon):
        k = st[i]; blk[i] = clock[k] // lw; clock[k] += 1
    return blk


def debounce(hp, mon, st, blk) -> np.ndarray:
    D = np.full(len(hp), -np.inf)
    ok = np.zeros(len(hp), bool)
    ok[2:] = (mon[2:] & mon[1:-1] & mon[:-2] & (st[2:] == st[1:-1]) & (st[1:-1] == st[:-2])
              & (blk[2:] == blk[1:-1]) & (blk[1:-1] == blk[:-2]))
    m3 = np.minimum(np.minimum(hp[2:], hp[1:-1]), hp[:-2])
    D[2:][ok[2:]] = m3[ok[2:]]
    return D


def block_scores(D, st, blk):
    """{k: array of block maxima S_{k,j}, j = 0..} (-inf if a block has no valid D)."""
    out = {}
    for k in range(3):
        sel = (st == k) & (blk >= 0)
        nb = blk[sel].max() + 1 if sel.any() else 0
        S = np.full(nb, -np.inf)
        np.maximum.at(S, blk[sel], D[sel])
        out[k] = S
    return out


def complete_blocks(mon, st, blk, lw: int = LW):
    """Number of COMPLETE blocks per state (the trailing partial block is not a calibration block)."""
    return {k: int(np.sum((st == k) & mon) // lw) for k in range(3)}


def main_thresholds(S, ncomp, n_cal=6, kstar=6, thin=1):
    """Per state: tau_k = kstar-th smallest of the first n_cal blocks (thinned by `thin`); None if fewer
    than n_cal complete blocks."""
    tau = {}
    for k in range(3):
        if ncomp[k] < n_cal:
            tau[k] = None; continue
        c = np.sort(S[k][:n_cal][::thin])
        tau[k] = float(c[kstar - 1])
    return tau


def test_mask(st, blk, tau, ncomp, n_cal=6, g=0):
    """Windows in COMPLETE test blocks of states whose threshold exists (R1.md: trailing incomplete block
    of each state is excluded, so monitored test hours = (L/60) x number of test blocks exactly)."""
    m = np.zeros(len(st), bool)
    for k in range(3):
        if tau.get(k) is not None:
            m |= (st == k) & (blk >= n_cal + g) & (blk < ncomp[k])
    return m


def alarms(D, st, blk, thr, tmask):
    """Alarm start indices. thr: per-window threshold array."""
    cand = np.flatnonzero(tmask & (D > thr))
    seen = set(); out = []
    for i in cand:
        key = (int(st[i]), int(blk[i]))
        if key not in seen:
            seen.add(key); out.append(int(i))
    return np.array(out, int)


def thr_from_state(st, per_state: dict) -> np.ndarray:
    return np.array([per_state.get(int(k)) if per_state.get(int(k)) is not None else np.inf for k in st], float)


def monitored_hours(mon, tmask) -> float:
    return float(np.sum(mon & tmask) * 2 / 3600)
