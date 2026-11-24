"""Per-window spectral features for E2-CNN (prereg v2.1 §3). Per-window normalisation only.

Window i covers [2i, 2i+8) s. Input signals must be PPG at 64 Hz and ACC (g) at 32 Hz.
Output X[n, 4, 256]: magnitude spectra on 256 linearly spaced frequencies in [0.5, 4] Hz of
PPG (normalised by its own max) and of ACC x/y/z (normalised jointly by the max over the 3 axes).
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt, detrend, resample_poly

FQ = np.linspace(0.5, 4.0, 256)
HR_GRID = np.arange(40, 201)            # 161 classes


def _windows(x, fs, n):
    w = sliding_window_view(x, 8 * fs, axis=0)[::2 * fs]
    return w[:n]


def _spec(W, fs, nfft):
    sos = butter(4, (0.5, 4.0), btype="band", fs=fs, output="sos")
    W = sosfiltfilt(sos, detrend(W, axis=-1), axis=-1) * np.hanning(W.shape[-1])
    P = np.abs(np.fft.rfft(W, nfft, axis=-1))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    idx = np.clip(np.searchsorted(f, FQ), 1, len(f) - 1)
    lo = f[idx - 1]; t = (FQ - lo) / (f[idx] - lo)
    return P[..., idx - 1] * (1 - t) + P[..., idx] * t


def spectra(ppg64, acc32, n_windows):
    n = min(n_windows, (len(ppg64) // 64 - 8) // 2 + 1, (len(acc32) // 32 - 8) // 2 + 1)
    P = _spec(_windows(ppg64, 64, n), 64, 2048)
    A = np.stack([_spec(_windows(acc32[:, k], 32, n), 32, 1024) for k in range(3)], 1)
    P = P / (P.max(-1, keepdims=True) + 1e-9)
    A = A / (A.max(axis=(1, 2), keepdims=True) + 1e-9)
    return np.concatenate([P[:, None, :], A], 1).astype(np.float32)


def to_dalia_rates(ppg, fs_ppg, acc, fs_acc):
    """Resample any device to DaLiA rates (PPG 64 Hz, ACC 32 Hz) with polyphase filtering."""
    from math import gcd
    g1 = gcd(64, fs_ppg); g2 = gcd(32, fs_acc)
    return (resample_poly(ppg, 64 // g1, fs_ppg // g1),
            resample_poly(acc, 32 // g2, fs_acc // g2, axis=0))
