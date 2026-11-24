"""E2-CNN (prereg v2.1 §3) and the CNN+HMM causal tracker."""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn
from .features import HR_GRID


class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, len(HR_GRID)))

    def forward(self, x):
        return self.net(x)


def soft_targets(hr, sigma=3.0):
    d = HR_GRID[None, :] - np.clip(hr, 40, 200)[:, None]
    t = np.exp(-0.5 * (d / sigma) ** 2)
    return (t / t.sum(1, keepdims=True)).astype(np.float32)


def train(X, y, seed, epochs=15, batch=256, lr=1e-3, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = SpecCNN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X); Tt = torch.from_numpy(soft_targets(y))
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            logp = torch.log_softmax(m(Xt[b].to(device)), 1)
            loss = -(Tt[b].to(device) * logp).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.cpu().eval()


@torch.no_grad()
def predict_proba(m, X, batch=4096):
    out = [torch.softmax(m(torch.from_numpy(X[i:i + batch])), 1).numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out)


def hmm_track(proba, sigma=8.0, floor=1e-3):
    d = np.arange(-30, 31, 1.0)
    k = np.exp(-0.5 * (d / sigma) ** 2); k /= k.sum()
    belief = np.full(proba.shape[1], 1.0 / proba.shape[1]); out = np.empty(len(proba))
    for i, p in enumerate(proba):
        prior = np.convolve(belief, k, mode="same") + floor / len(belief)
        post = prior * (p + 1e-6); belief = post / post.sum()
        out[i] = HR_GRID[np.argmax(belief)]
    return out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------------------------
# v2.1 deviation #5 (dev finding, S1c): SpecCNN's global average pooling discards WHERE the spectral peak
# is, so it regresses to activity-typical HR (pred range 50-115 bpm; injected +30 bpm recovered ~1 bpm).
# SpecFCN is frequency-equivariant: one logit per input frequency bin (classes = bins within 40-200 bpm).
from .features import FQ
FCN_MASK = (FQ * 60 >= 40) & (FQ * 60 <= 200)
FCN_GRID = FQ[FCN_MASK] * 60.0                      # ~0.82 bpm spacing


class SpecFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 7, padding=6, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 1, 7, padding=3))
        self.register_buffer("mask", torch.from_numpy(FCN_MASK))

    def forward(self, x):
        return self.net(x)[:, 0, :][:, self.mask]


def soft_targets_grid(hr, grid, sigma=3.0):
    d = grid[None, :] - np.clip(hr, grid[0], grid[-1])[:, None]
    t = np.exp(-0.5 * (d / sigma) ** 2)
    return (t / t.sum(1, keepdims=True)).astype(np.float32)


def train_fcn(X, y, seed, epochs=15, batch=256, lr=1e-3, init=None):
    """init: optional state_dict to start from (fine-tuning); None = fresh model (all results before 22-11)."""
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = SpecFCN()
    if init is not None:
        m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X); Tt = torch.from_numpy(soft_targets_grid(y, FCN_GRID))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            loss = -(Tt[b] * torch.log_softmax(m(Xt[b]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def hmm_track_grid(proba, grid, sigma_bpm=8.0, floor=1e-3):
    step = float(np.mean(np.diff(grid)))
    d = np.arange(-40, 41) * step
    k = np.exp(-0.5 * (d / sigma_bpm) ** 2); k /= k.sum()
    belief = np.full(proba.shape[1], 1.0 / proba.shape[1]); out = np.empty(len(proba))
    for i, p in enumerate(proba):
        prior = np.convolve(belief, k, mode="same") + floor / len(belief)
        post = prior * (p + 1e-6); belief = post / post.sum()
        out[i] = grid[np.argmax(belief)]
    return out
