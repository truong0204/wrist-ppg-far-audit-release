"""E1: spectral-peak HR estimator with accelerometer-based motion-peak removal and causal tracking.

Deterministic, no learned parameters, MCU-friendly. All parameters are fixed in PARAMS and were set
before any H32 data was processed (prereg §P3); they are NOT tuned on pilot or test data.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks

PARAMS = dict(
    win_s=8, shift_s=2,
    band_bpm=(40.0, 200.0),      # search band
    nfft_bvp=8192, nfft_acc=4096,  # 0.47 bpm bin at both rates
    bp_hz=(0.5, 4.0), bp_order=4,
    motion_std_g=0.02,           # window is "in motion" if std of |acc| (g) exceeds this
    acc_peak_rel=0.5,            # ACC spectral peaks >= 0.5 * max are motion frequencies
    n_acc_peaks=2,
    mask_halfwidth_bpm=6.0,      # remove PPG power within +-6 bpm of motion fundamental and 2nd harmonic
    track_bpm=15.0,              # next estimate must lie within +-15 bpm of previous
    max_hold=5,                  # after 5 held windows (10 s) fall back to the global maximum
)


def _spec(x, fs, nfft, sos):
    x = sosfiltfilt(sos, detrend(x))
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    return np.fft.rfftfreq(nfft, 1.0 / fs) * 60.0, p


def acc_in_g(acc: np.ndarray) -> np.ndarray:
    """E4 raw ACC is in 1/64 g (DaLiA README III.2). Auto-detect: at rest |acc| ~ 64 vs ~ 1."""
    mag = np.median(np.linalg.norm(acc, axis=1))
    return acc / 64.0 if mag > 10 else acc


def estimate(bvp: np.ndarray, acc: np.ndarray, n_windows: int, fs_bvp=64, fs_acc=32, p=PARAMS):
    acc = acc_in_g(acc)
    sos_b = butter(p["bp_order"], p["bp_hz"], btype="band", fs=fs_bvp, output="sos")
    sos_a = butter(p["bp_order"], p["bp_hz"], btype="band", fs=fs_acc, output="sos")
    lo, hi = p["band_bpm"]
    hr = np.full(n_windows, np.nan); motion = np.zeros(n_windows, bool)
    prev, held = None, 0
    for i in range(n_windows):
        b = bvp[i * p["shift_s"] * fs_bvp:(i * p["shift_s"] + p["win_s"]) * fs_bvp]
        a = acc[i * p["shift_s"] * fs_acc:(i * p["shift_s"] + p["win_s"]) * fs_acc]
        if len(b) < p["win_s"] * fs_bvp or len(a) < p["win_s"] * fs_acc:
            hr[i] = prev if prev is not None else np.nan; continue
        f, pp = _spec(b, fs_bvp, p["nfft_bvp"], sos_b)
        band = (f >= lo) & (f <= hi)
        f, pp = f[band], pp[band].copy()
        motion[i] = np.std(np.linalg.norm(a, axis=1)) > p["motion_std_g"]
        if motion[i]:
            fa = None; pa = 0
            for ax in range(3):
                fa_, pa_ = _spec(a[:, ax], fs_acc, p["nfft_acc"], sos_a)
                pa = pa + pa_; fa = fa_
            sel = (fa >= lo) & (fa <= hi)
            fa, pa = fa[sel], pa[sel]
            pk, _ = find_peaks(pa, height=p["acc_peak_rel"] * pa.max())
            for j in pk[np.argsort(pa[pk])[::-1][:p["n_acc_peaks"]]]:
                for h in (1, 2):
                    pp[np.abs(f - h * fa[j]) <= p["mask_halfwidth_bpm"]] = 0.0
        pk, _ = find_peaks(pp)
        if pk.size == 0:
            hr[i] = prev if prev is not None else np.nan; held += 1; continue
        if prev is None or held >= p["max_hold"]:
            est = f[pk[np.argmax(pp[pk])]]; held = 0
        else:
            near = pk[np.abs(f[pk] - prev) <= p["track_bpm"]]
            if near.size:
                est = f[near[np.argmax(pp[near])]]; held = 0
            else:
                est = prev; held += 1
        hr[i] = prev = est
    return hr, motion
