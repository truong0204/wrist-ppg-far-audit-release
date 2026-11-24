"""M3 (Cowork r1): confirmatory statistics at the aggregate level and their power, by simulation. DEV-derived
design only: the number of test blocks per (person, state) cell is taken from the 3 WildPPG dev IDs
(02_data/derived/s2_cache, seed 0, sigma 3, main method n_cal = 6, k* = 6).

Null H0 (R1.md Prop. 2a under A(0)): per cell, p_c ~ Beta(1, 6) independently; N_c | p_c ~ Bin(T_c, p_c);
observed rate = (60/L) N_c / T_c; E_c = 1{observed rate > r}, r = 2/h.
Statistics (13 persons x 3 states, cells without a within-person threshold are absent):
  X  = sum E_c                          test X1: X > q95 of its exact null (Poisson-binomial, independent cells)
                                        test X2: person-cluster bootstrap, lower 95 % bound of X - E0[X] > 0
  Y  = sum N_c (total false alarms)     test Y1: Y > q95 of its null (independent cells, simulated)
                                        test Y2: person-cluster bootstrap, lower 95 % bound of Y - E0[Y] > 0
Alternatives: a fraction f in {0.2, 0.4} of cells have p = c alpha with c in {1.5, 2} (true rate 3 or 4 /h).
Also f = 0.2/0.4 with c = 1.25. Seeds 0-4 x 2,000 replicates; bootstrap 500.
    uv run python 05_experiments/s2/m3_power.py
"""
import sys, json, pathlib, numpy as np
from scipy.stats import betabinom
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; L = 10; r = 2.0; alpha = r * L / 60; NP = 13; REPS = 2000; NB = 500

prof = []
for sid in ("e61", "u7y", "ngh"):
    z = np.load(C / f"WildPPG_{sid}_seed0.npz"); st = z["st"]; mon = z["sq"] >= TH
    blk = M.blocks(mon, st); nc = M.complete_blocks(mon, st, blk)
    prof.append([max(nc[k] - 6, 0) for k in range(3)])
print("test blocks per state (rest, light, vigorous) of dev IDs:", prof)


def p0_exceed(T):
    """P0(observed rate > r) for a cell with T test blocks: N > T alpha... rate = 6 N / T > 2 <=> N > T / 3."""
    n = np.arange(T + 1); pm = betabinom.pmf(n, T, 1, 6)
    return float(pm[(60 / L) * n / T > r].sum()), float((pm * n).sum())


def simulate(f, c, rng):
    rej = dict(X1=0, X2=0, Y1=0, Y2=0); Xs = []
    for _ in range(REPS):
        Ts = np.array([prof[i] for i in rng.integers(0, 3, NP)])                # persons x states
        cells = [(i, k, int(Ts[i, k])) for i in range(NP) for k in range(3) if Ts[i, k] > 0]
        P0 = np.array([p0_exceed(T)[0] for _, _, T in cells]); EN = np.array([p0_exceed(T)[1] for _, _, T in cells])
        bad = rng.random(len(cells)) < f
        p = np.where(bad, c * alpha, rng.beta(1, 6, len(cells)))
        N = rng.binomial([T for _, _, T in cells], np.minimum(p, 1))
        T_ = np.array([T for _, _, T in cells]); E = ((60 / L) * N / T_ > r).astype(float)
        X, Y = E.sum(), N.sum(); Xs.append(X)
        # exact-ish nulls by simulation under independence (cheap)
        p_null = rng.beta(1, 6, (400, len(cells))); N_null = rng.binomial(T_, p_null)
        X_null = ((60 / L) * N_null / T_ > r).sum(1); Y_null = N_null.sum(1)
        rej["X1"] += X > np.quantile(X_null, 0.95); rej["Y1"] += Y > np.quantile(Y_null, 0.95)
        pid = np.array([i for i, _, _ in cells]); dX = np.bincount(pid, E - P0, NP); dY = np.bincount(pid, N - EN, NP)
        bi = rng.integers(0, NP, (NB, NP))
        rej["X2"] += np.quantile(dX[bi].sum(1), 0.05) > 0; rej["Y2"] += np.quantile(dY[bi].sum(1), 0.05) > 0
    return {k: v / REPS for k, v in rej.items()}, float(np.mean(Xs))


res = []
for f, c in [(0.0, 1.0), (0.2, 1.25), (0.4, 1.25), (0.2, 1.5), (0.4, 1.5), (0.2, 2.0), (0.4, 2.0)]:
    pw = [simulate(f, c, np.random.default_rng(s)) for s in range(5)]
    row = dict(frac_bad=f, mult=c, **{k: float(np.mean([p[0][k] for p in pw])) for k in ("X1", "X2", "Y1", "Y2")},
               sd={k: float(np.std([p[0][k] for p in pw])) for k in ("X1", "X2", "Y1", "Y2")}, mean_X=float(np.mean([p[1] for p in pw])))
    res.append(row)
    print("f=%.1f c=%.2f | power X1 %.2f X2 %.2f | Y1 %.2f Y2 %.2f | mean X %.1f" % (f, c, row["X1"], row["X2"], row["Y1"], row["Y2"], row["mean_X"]), flush=True)
json.dump(dict(profiles=prof, n_persons=NP, results=res), open(OUT / "m3_power.json", "w"), indent=1)
