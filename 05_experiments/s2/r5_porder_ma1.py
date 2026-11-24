"""Rev 5 item 1 (PI): size of p_order under STATIONARY serial dependence (no drift), 13 persons.
Block scores per cell: Gaussian MA(1), X_j = (e_j + theta e_{j-1}) / sqrt(1 + theta^2), with lag-1 autocorrelation
rho1 = theta / (1 + theta^2) in {0 (iid), 0.17 (dev pooled detrended), 0.30 (dev ngh)}. Number of complete blocks per
cell from the 3 dev persons (seed 0), persons drawn with replacement. p_order exactly as prereg rev 4 §4.2
(K = max of blocks 0..5, test blocks >= 6, X = #cells with 6 N / T > 2; oracle = 6 random calibration blocks per
cell, B = 999; p = (1 + #{X_oracle >= X_K}) / (B + 1); reject if p < 0.05).
Also reported for information: X2 (H-K) under K (m = 0), and the exceedance of K-m1 (thinned calibration: blocks
0, 2, ..., 10; buffer 1; test blocks >= 13) as mean rate, to show which analysis keeps its level under MA(1).
Stop rule (PI): if size(p_order) at rho1 = 0.17 exceeds 0.10 -> STOP and report, no design change.
5 seeds x 1,000 replicates.
    uv run python 05_experiments/s2/r5_porder_ma1.py
"""
import sys, json, pathlib, numpy as np
from scipy.stats import betabinom
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; NC = 6; R_ = 2.0; B = 999

PROF = []
for sid in ("e61", "u7y", "ngh"):
    z = np.load(C / f"WildPPG_{sid}_seed0.npz"); st = z["st"]; mon = z["sq"] >= TH; blk = M.blocks(mon, st)
    nc = M.complete_blocks(mon, st, blk); PROF.append([nc[k] for k in range(3)])


def theta_for(rho):
    return 0.0 if rho == 0 else (1 - np.sqrt(1 - 4 * rho ** 2)) / (2 * rho)


P0 = {}
def p0(T):
    if T not in P0:
        n = np.arange(T + 1); P0[T] = float(betabinom.pmf(n, T, 1, 6)[6 * n / T > R_].sum())
    return P0[T]


def rep(g, th):
    XK = 0; Xo = np.zeros(B); d = np.zeros(13); m1_N = m1_T = 0; k_N = k_T = 0
    for i in range(13):
        for n in PROF[g.integers(0, 3)]:
            if n < NC + 1: continue
            e = g.standard_normal(n + 1); S = (e[1:] + th * e[:-1]) / np.sqrt(1 + th ** 2); T = n - NC
            N = np.sum(S[NC:] > S[:NC].max()); ex = 6 * N / T > R_; XK += ex; d[i] += ex - p0(T); k_N += N; k_T += T
            keys = g.random((B, n)); cal = np.argpartition(keys, NC - 1, axis=1)[:, :NC]
            Xo += (6 * (S[None, :] > S[cal].max(1)[:, None]).sum(1) / T) > R_
            if n >= 14:
                tau1 = S[0:12:2].max(); m1_N += np.sum(S[13:] > tau1); m1_T += n - 13
    p = (1 + np.sum(Xo >= XK)) / (B + 1)
    bi = g.integers(0, 13, (500, 13))
    return p < 0.05, np.quantile(d[bi].sum(1), 0.05) > 0, 6 * k_N / k_T, (6 * m1_N / m1_T if m1_T else np.nan)


res = {}
for rho in (0.0, 0.17, 0.30):
    th = theta_for(rho); r_po, r_x2, kr, m1r = [], [], [], []
    for seed in range(5):
        g = np.random.default_rng(700 + seed); rr = [rep(g, th) for _ in range(1000)]
        r_po.append(np.mean([a for a, _, _, _ in rr])); r_x2.append(np.mean([b for _, b, _, _ in rr]))
        kr.append(np.mean([c for _, _, c, _ in rr])); m1r.append(np.nanmean([x for _, _, _, x in rr]))
    res[f"rho1_{rho}"] = dict(theta=float(th), size_p_order=float(np.mean(r_po)), sd_p_order=float(np.std(r_po)),
                             size_X2_K=float(np.mean(r_x2)), mean_rate_K=float(np.mean(kr)), mean_rate_K_m1=float(np.mean(m1r)))
    print(rho, json.dumps(res[f"rho1_{rho}"]), flush=True)
stop = res["rho1_0.17"]["size_p_order"] > 0.10
res["stop_rule"] = dict(threshold=0.10, size_at_0_17=res["rho1_0.17"]["size_p_order"], STOP=bool(stop))
res["profiles"] = PROF
json.dump(res, open(OUT / "r5_porder_ma1.json", "w"), indent=1)
print("STOP" if stop else "continue", res["stop_rule"])
