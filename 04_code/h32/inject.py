"""Semi-synthetic tachycardia injection (prereg v2.1 §2).

Inside the segment [t0, t1) the PPG is read faster: y(t) = x(tau(t)), dtau/dt = (HR(t)+D(t))/HR(t), so the pulse
rate (and its harmonics) rises by D(t). D(t) ramps 0 -> delta over `ramp_s`, holds for `plateau_s`, ramps back
over `ramp_s`. After t1 the signal returns to the original time base (alignment with ACC and reference kept)
through a 2 s cross-fade. The reference HR per 2 s window is raised by D at the window centre.
Limitations (stated in the paper): motion artefacts inside the segment are time-compressed while ACC is not;
one cross-fade window per event.
"""
from __future__ import annotations
import numpy as np


def profile(t, t0, ramp_s, plateau_s, delta):
    """Offset D(t) in bpm."""
    a = np.clip((t - t0) / ramp_s, 0, 1)
    b = np.clip((t0 + 2 * ramp_s + plateau_s - t) / ramp_s, 0, 1)
    return delta * np.minimum(a, b)


def inject(ppg, fs, hr_ref_2s, t0, delta, ramp_s=30.0, plateau_s=60.0, xfade_s=2.0):
    """Return (ppg_new, hr_ref_new, (t_start, t_end)). hr_ref_2s: reference per 2 s window (window i centre 2i+4 s)."""
    ppg = np.asarray(ppg, float).copy()
    t1 = t0 + 2 * ramp_s + plateau_s
    i0, i1 = int(t0 * fs), min(int(t1 * fs), len(ppg))
    t = np.arange(i0, i1) / fs
    centres = np.arange(len(hr_ref_2s)) * 2 + 4.0
    hr_t = np.interp(t, centres, np.nan_to_num(hr_ref_2s, nan=np.nanmedian(hr_ref_2s)))
    rate = (hr_t + profile(t, t0, ramp_s, plateau_s, delta)) / np.maximum(hr_t, 30.0)
    tau = t0 + np.concatenate([[0.0], np.cumsum(rate[:-1]) / fs])
    src = np.arange(len(ppg)) / fs
    seg = np.interp(tau, src, ppg)
    ppg[i0:i1] = seg
    nx = int(xfade_s * fs)
    if i1 + nx <= len(ppg):                         # cross-fade from warped continuation back to original
        tau_end = tau[-1] + rate[-1] / fs
        cont = np.interp(tau_end + np.arange(nx) / fs, src, ppg)
        w = np.linspace(1, 0, nx)
        ppg[i1:i1 + nx] = w * cont + (1 - w) * ppg[i1:i1 + nx]
    hr_new = np.asarray(hr_ref_2s, float) + profile(centres, t0, ramp_s, plateau_s, delta)
    return ppg, hr_new, (t0, t1)


def schedule(n_windows, eligible, rng, per_hour=3, min_gap_s=600, dur_s=180.0, start_s=0.0):
    """Random onsets (s) among eligible 2 s windows (bool array), >= min_gap_s apart, <= per_hour per hour."""
    total_h = n_windows * 2 / 3600
    k = int(np.floor(per_hour * total_h))
    cand = np.flatnonzero(eligible) * 2.0
    cand = cand[(cand >= start_s) & (cand + dur_s < n_windows * 2)]
    rng.shuffle(cand)
    out = []
    for c in cand:
        if all(abs(c - o) >= min_gap_s for o in out):
            out.append(float(c))
        if len(out) >= k:
            break
    return sorted(out)
