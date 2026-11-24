"""Episode definitions and alarm-to-reference matching (window i covers [2i, 2i+8) s)."""
from __future__ import annotations
import numpy as np


def episodes(flag: np.ndarray, k_min: int, merge_gap: int) -> list[tuple[int, int]]:
    """Runs of True; runs separated by <= merge_gap False windows are merged; then episodes
    spanning fewer than k_min windows are dropped. Returns inclusive (start, end) window indices."""
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    eps, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e - 1 <= merge_gap:
            e = i
        else:
            eps.append((s, e)); s = e = i
    eps.append((s, e))
    return [(s, e) for s, e in eps if e - s + 1 >= k_min]


def to_seconds(ep, shift_s=2, win_s=8):
    s, e = ep
    return s * shift_s, e * shift_s + win_s


def match(alarm_eps, ref_eps, tol_s: float):
    """False alarm = alarm episode not overlapping any reference episode widened by tol_s.
    A reference episode is detected if some alarm episode overlaps it (same widening)."""
    A = [to_seconds(a) for a in alarm_eps]
    R = [(r0 - tol_s, r1 + tol_s) for r0, r1 in (to_seconds(r) for r in ref_eps)]
    ov = lambda x, y: x[0] < y[1] and y[0] < x[1]
    fa = [a for a in A if not any(ov(a, r) for r in R)]
    det = [r for r in R if any(ov(a, r) for a in A)]
    return len(fa), len(det)
