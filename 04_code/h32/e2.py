"""E2: causal Bayesian (HMM forward-filter) HR tracker on a 1-bpm grid.

Per 8 s window: PPG power spectrum (same front end as E1) on a 40–200 bpm grid, accelerometer spectrum
used to attenuate motion frequencies, plus 2nd-harmonic support; then a forward filter with a Gaussian
random-walk prior on HR (sigma bpm per 2 s step) and a small uniform floor so the tracker can recover.
Strictly causal (window t uses windows <= t). State size 161; per step O(161 x kernel), MCU-feasible.
Tuned parameters (sigma, w, gamma) are chosen on development subjects only (prereg_v2 §5).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend
from .e1 import acc_in_g

GRID = np.arange(40.0, 201.0, 1.0)
DEFAULT = dict(sigma=4.0, w=1.0, gamma=2.0, harm=0.5, floor=1e-3, motion_std_g=0.02,
               win_s=8, shift_s=2, nfft_bvp=8192, nfft_acc=4096, bp_hz=(0.5, 4.0), bp_order=4)


def _spec_on(x, fs, nfft, sos, grid):
    x = sosfiltfilt(sos, detrend(x))
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs) * 60.0
    return np.interp(grid, f, p)


def estimate(bvp, acc, n_windows, fs_bvp=64, fs_acc=32, p=None):
    p = {**DEFAULT, **(p or {})}
    acc = acc_in_g(acc)
    sos_b = butter(p["bp_order"], p["bp_hz"], btype="band", fs=fs_bvp, output="sos")
    sos_a = butter(p["bp_order"], p["bp_hz"], btype="band", fs=fs_acc, output="sos")
    d = np.arange(-30, 31, 1.0)
    kern = np.exp(-0.5 * (d / p["sigma"]) ** 2); kern /= kern.sum()
    grid2 = 2 * GRID
    belief = np.full(GRID.size, 1.0 / GRID.size)
    hr = np.full(n_windows, np.nan)
    W, S = p["win_s"], p["shift_s"]
    for i in range(n_windows):
        b = bvp[i * S * fs_bvp:(i * S + W) * fs_bvp]
        a = acc[i * S * fs_acc:(i * S + W) * fs_acc]
        if len(b) < W * fs_bvp or len(a) < W * fs_acc:
            hr[i] = hr[i - 1] if i else np.nan; continue
        P = _spec_on(b, fs_bvp, p["nfft_bvp"], sos_b, GRID)
        P2 = _spec_on(b, fs_bvp, p["nfft_bvp"], sos_b, grid2)   # power at 2f (band-pass rolls off above 240 bpm)
        scale = P.max() + 1e-12                                   # normalise both by the SAME fundamental max
        P = P / scale; P2 = P2 / scale
        if np.std(np.linalg.norm(a, axis=1)) > p["motion_std_g"]:
            A = sum(_spec_on(a[:, k], fs_acc, p["nfft_acc"], sos_a, GRID) for k in range(3))
            A = A / (A.max() + 1e-12)
            P = np.clip(P * (1 - p["w"] * A), 0, None)
        L = (P + p["harm"] * np.clip(P2, 0, None) + 1e-6) ** p["gamma"]
        prior = np.convolve(belief, kern, mode="same") + p["floor"] / GRID.size
        post = prior * L
        belief = post / post.sum()
        hr[i] = GRID[np.argmax(belief)]
    return hr
