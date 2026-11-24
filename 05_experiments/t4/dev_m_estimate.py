"""Dev-only check of assumption A(m) (03_theory/R1.md): autocorrelation of state-time block scores.
WildPPG dev e61/u7y/ngh (zero-shot all-DaLiA SpecFCN) and DaLiA dev S10-S12 (LOSO fold models); seeds 0-4;
E2 = SpecFCN + HMM sigma 3; SQI theta = 0.02896; states v2 §3; L = 10 min (300 windows on the state clock);
D_i = min of 3 real-time-consecutive windows in the same state-time block; S = max D per block.
Rule fixed before running: m = smallest m >= 0 such that the pooled lag-l autocorrelation (blocks of the same
state, pooled over IDs/states/seeds, weighted by number of pairs) is < 0.20 for every l in m+1..3.
    uv run python 05_experiments/t4/dev_m_estimate.py
"""
import sys, json, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W, io_dalia, e2cnn, features
torch.set_num_threads(8)
R = ROOT / "05_experiments/s1c/results"; GRID = e2cnn.FCN_GRID; TH = 0.02896360121667385; SIG = 3.0; LW = 300


def block_scores(hp, sq, acc32, n):
    s = np.array([np.std(np.linalg.norm(acc32[i * 64:i * 64 + 256], axis=1)) for i in range(n)])
    st = np.digitize(s, [0.02, 0.10]); mon = sq[:n] >= TH
    clock = np.zeros(3, int); blk = np.full(n, -1)
    for i in range(n):
        if mon[i]:
            blk[i] = clock[st[i]] // LW; clock[st[i]] += 1
    out = {k: {} for k in range(3)}
    for i in range(2, n):
        if mon[i] and mon[i - 1] and mon[i - 2] and st[i] == st[i - 1] == st[i - 2] and blk[i] == blk[i - 1] == blk[i - 2]:
            d = min(hp[i], hp[i - 1], hp[i - 2]); k = st[i]
            out[k][blk[i]] = max(out[k].get(blk[i], -np.inf), d)
    return {k: np.array([v[j] for j in sorted(v)]) for k, v in out.items()}


def acf(x, l):
    if len(x) <= l + 2: return None, 0
    x = x - x.mean(); den = np.sum(x * x)
    return (float(np.sum(x[:-l] * x[l:]) / den) if den > 0 else None), len(x) - l


rows = []
def add(src, sid, k_seed, S):
    for k, x in S.items():
        for l in (1, 2, 3):
            r, npair = acf(x, l)
            if r is not None:
                rows.append(dict(src=src, sid=sid, seed=k_seed, state=int(k), lag=l, acf=r, pairs=npair, n_blocks=len(x)))

for s in W.DEV:
    d = W.load_wrist(s); p64, a32 = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
    n = (len(p64) // 64 - 8) // 2 + 1; X = features.spectra(p64, a32, n)
    for k in SEEDS if (SEEDS := [0, 1, 2, 3, 4]) else []:
        m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(R / f"fcn_all_seed{k}.pt")); m.eval()
        P = e2cnn.predict_proba(m, X); add("WildPPG", s, k, block_scores(e2cnn.hmm_track_grid(P, GRID, SIG), P.max(1), a32, len(P)))
    print(s, flush=True)
Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz")
for s in ("S10", "S11", "S12"):
    dd = io_dalia.load_subject(s, need_ppg=False)
    for k in SEEDS:
        P = np.load(R / f"fcn_proba_{s}_seed{k}.npy")
        add("DaLiA", s, k, block_scores(e2cnn.hmm_track_grid(P, GRID, SIG), P.max(1), dd["acc"], len(P)))
    print(s, flush=True)

summ = {}
for src in ("WildPPG", "DaLiA", "all"):
    sel = [r for r in rows if src == "all" or r["src"] == src]
    summ[src] = {l: float(np.average([r["acf"] for r in sel if r["lag"] == l], weights=[r["pairs"] for r in sel if r["lag"] == l])) for l in (1, 2, 3)}
    blocks = {st: int(np.mean([r["n_blocks"] for r in sel if r["lag"] == 1 and r["state"] == st] or [0])) for st in range(3)}
    print(src, "pooled acf lag1..3:", {l: round(v, 3) for l, v in summ[src].items()}, "| mean blocks per (ID, state):", blocks)
m_hat = next(m for m in range(4) if all(summ["all"][l] < 0.20 for l in range(m + 1, 4)))
print("m_hat =", m_hat, "(rule: pooled acf < 0.20 for all lags > m)")
json.dump(dict(rows=rows, pooled=summ, m_hat=m_hat), open(ROOT / "05_experiments/t4/results/dev_m_estimate.json", "w"), indent=1)
