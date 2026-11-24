"""Rev 6 (PI): H-K1 = second test of a fixed sequence (tested only if H-K is rejected; each at 5 %, no alpha split).
K with m = 1: tau_c = max of blocks 0, 2, ..., 10 (thinned calibration from n_cal = 12), buffer 1, test blocks >= 13;
cells with >= 14 complete blocks. Statistic: d_i = (seed mean of) sum over the person's cells of (N_c - T_c / 7);
reject iff the 5 % quantile of the person-cluster bootstrap of sum_i d_i is > 0. Under stationary 1-dependence
E[N_c] <= T_c / 7 holds (the 7 variables are independent), although N_c is no longer BetaBinomial.

Part A (simulation, 13 persons, block profiles of the 3 dev persons, Gaussian blocks, 5 seeds x 1,000 replicates,
bootstrap 500): size under iid and stationary MA(1) with lag-1 0.17 / 0.30; power under linear drift of the dev size
(0.14 SD per block, ngh vigorous) in all cells, in half of the cells, and 0.07 SD in all cells. H-K (X2, K m = 0,
P0 from BetaBinomial(T, 1, 6)) is computed on the same data: reported are P(H-K1 rejects) and the fixed-sequence
power P(H-K and H-K1 both reject).
Stop rule (PI): if the size of H-K1 under MA(1) lag-1 0.30 exceeds 0.07 -> STOP and report.

Part B (dev, WildPPG e61/u7y/ngh, sigma 3, seeds 0-4): K m = 0 vs K m = 1 on the SAME test blocks (>= 13), per cell,
per person and mean of persons; dev value of the H-K1 statistic (3-cluster bootstrap: descriptive).
    uv run python 05_experiments/s2/r6_hk1.py
"""
import sys, json, pathlib, itertools, numpy as np
from scipy.stats import betabinom
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"; TH = 0.02896360121667385
IDS = ("e61", "u7y", "ngh"); STN = ["rest", "light", "vigorous"]; out = {}


def seqs(sid, s):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; mon = z["sq"] >= TH; blk = M.blocks(mon, st)
    D = M.debounce(z["hp3"], mon, st, blk); Sb = M.block_scores(D, st, blk); nc = M.complete_blocks(mon, st, blk)
    return [Sb[k][:nc[k]].astype(float) for k in range(3)]


PROF = [[len(S) for S in seqs(sid, 0)] for sid in IDS]
P0 = {}
def p0(T):
    if T not in P0:
        n = np.arange(T + 1); P0[T] = float(betabinom.pmf(n, T, 1, 6)[6 * n / T > 2].sum())
    return P0[T]


def stats_person(cells):
    """cells: list of S arrays of one person -> (d_HK, d_HK1)."""
    dk = dk1 = 0.0
    for S in cells:
        n = len(S)
        if n >= 7:
            T = n - 6; N = np.sum(S[6:] > S[:6].max()); dk += (6 * N / T > 2) - p0(T)
        if n >= 14:
            T1 = n - 13; N1 = np.sum(S[13:] > S[0:12:2].max()); dk1 += N1 - T1 / 7
    return dk, dk1


def boot_reject(d, g, B=500):
    bi = g.integers(0, len(d), (B, len(d))); return np.quantile(d[bi].sum(1), 0.05) > 0


def gen(g, n, model, par):
    if model == "ma1":
        th = 0.0 if par == 0 else (1 - np.sqrt(1 - 4 * par ** 2)) / (2 * par)
        e = g.standard_normal(n + 1); return (e[1:] + th * e[:-1]) / np.sqrt(1 + th ** 2)
    return g.standard_normal(n) + par * np.arange(n)


sim = {}
for name, model, par, frac in [("iid", "ma1", 0.0, 1), ("MA1_lag1_0.17", "ma1", 0.17, 1), ("MA1_lag1_0.30", "ma1", 0.30, 1),
                               ("drift_0.14_all", "drift", 0.14, 1), ("drift_0.14_half", "drift", 0.14, 0.5), ("drift_0.07_all", "drift", 0.07, 1)]:
    r1, rs, rk = [], [], []
    for seed in range(5):
        g = np.random.default_rng(1300 + seed); a1 = ak = aseq = 0
        for _ in range(1000):
            dk, dk1 = np.zeros(13), np.zeros(13)
            for i in range(13):
                cells = []
                for n in PROF[g.integers(0, 3)]:
                    use = frac >= 1 or g.random() < frac
                    cells.append(gen(g, n, model, par) if use else gen(g, n, "ma1", 0.0))
                dk[i], dk1[i] = stats_person(cells)
            hk = boot_reject(dk, g); hk1 = boot_reject(dk1, g)
            a1 += hk1; ak += hk; aseq += hk and hk1
        r1.append(a1 / 1000); rk.append(ak / 1000); rs.append(aseq / 1000)
    sim[name] = dict(P_HK1=float(np.mean(r1)), sd_HK1=float(np.std(r1)), P_HK=float(np.mean(rk)), P_sequence=float(np.mean(rs)))
    print("A", name, json.dumps(sim[name]), flush=True)
stop = sim["MA1_lag1_0.30"]["P_HK1"] > 0.07
out["simulation"] = sim; out["stop_rule"] = dict(threshold=0.07, size_at_0_30=sim["MA1_lag1_0.30"]["P_HK1"], STOP=bool(stop))
print("STOP" if stop else "continue", out["stop_rule"], flush=True)

# ---------------------------------------------------------------- Part B dev, paired on blocks >= 13
cells, dper = {}, {sid: [] for sid in IDS}
for sid, s in itertools.product(IDS, range(5)):
    d1 = 0.0
    for k, S in enumerate(seqs(sid, s)):
        if len(S) < 14: continue
        T = len(S) - 13; t0, t1 = S[:6].max(), S[0:12:2].max()
        N0, N1 = np.sum(S[13:] > t0), np.sum(S[13:] > t1); d1 += N1 - T / 7
        cells.setdefault((sid, STN[k]), []).append(dict(T=T, N0=int(N0), N1=int(N1), same_tau=bool(t0 == t1)))
    dper[sid].append(d1)
rows = []
for (sid, stn), v in cells.items():
    rows.append(dict(sid=sid, state=stn, T=float(np.mean([x["T"] for x in v])), K_m0=float(np.mean([6 * x["N0"] / x["T"] for x in v])),
                     K_m1=float(np.mean([6 * x["N1"] / x["T"] for x in v])), seeds_same_tau=int(sum(x["same_tau"] for x in v))))
pp = {}
for sid in IDS:
    N0 = np.mean([sum(cells[key][s]["N0"] for key in cells if key[0] == sid) for s in range(5)])
    N1 = np.mean([sum(cells[key][s]["N1"] for key in cells if key[0] == sid) for s in range(5)])
    T = np.mean([sum(cells[key][s]["T"] for key in cells if key[0] == sid) for s in range(5)])
    pp[sid] = dict(K_m0=float(6 * N0 / T), K_m1=float(6 * N1 / T), diff=float(6 * (N0 - N1) / T), test_h=float(T / 6))
dvec = np.array([np.mean(dper[sid]) for sid in IDS]); g = np.random.default_rng(0); bi = g.integers(0, 3, (10000, 3))
out["dev"] = dict(cells=rows, per_person=pp, mean_of_persons=dict(K_m0=float(np.mean([pp[s]["K_m0"] for s in IDS])), K_m1=float(np.mean([pp[s]["K_m1"] for s in IDS])),
                  diff=float(np.mean([pp[s]["diff"] for s in IDS]))),
                  HK1_d_i={s: float(d) for s, d in zip(IDS, dvec)}, HK1_sum_d=float(dvec.sum()), HK1_boot_q05=float(np.quantile(dvec[bi].sum(1), 0.05)))
for r in rows: print("B", r)
print("B per person", json.dumps(pp)); print("B H-K1 dev", out["dev"]["HK1_d_i"], out["dev"]["HK1_sum_d"], out["dev"]["HK1_boot_q05"])
json.dump(out, open(OUT / "r6_hk1.json", "w"), indent=1)
