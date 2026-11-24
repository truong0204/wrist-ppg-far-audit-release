"""PI prompt after S1f, part B (dev + simulation). WildPPG e61/u7y/ngh, sigma 3, E2 seeds 0-4.

B1  Paired R vs R-thin on the SAME test blocks (blocks j >= 13 of each cell): person rate 6 sum N / sum T,
    seed-averaged; difference R - R-thin per person and mean over persons.
B2  Lag-1 and lag-2 autocorrelation of block scores per cell (all complete blocks), raw and after removing a linear
    trend; Fisher z per cell, pooled per person with weights n - 3, then over persons; 90 % CI from the pooled
    Fisher SE 1/sqrt(sum(n - 3)). Seed-averaged z. NOTE: a trend inflates the raw acf, so raw lag-1 cannot by itself
    separate short-range dependence from drift; the detrended acf is the cleaner check (slightly biased downwards).
B3  Power of the one-sided one-sample t test of H-R (mean of person rates > 6/7) with 13 persons, block profiles
    from the 3 dev persons, Gaussian block scores:
    (i) MA(1), theta 0.5 and 1.0;  (ii) linear drift beta_z * j in SD units per block, beta_z = 0.14 (ngh vigorous:
    slope 2.8 bpm/block over residual SD 20.6 bpm), in all cells (common drift, L1) and in half the cells.
    Also: mean paired difference R - R-thin on the R-thin test blocks (directional prediction check).
    5 seeds x 4,000 replicates.
    uv run python 05_experiments/s2/b_rchecks.py
"""
import sys, json, pathlib, itertools, numpy as np
from scipy.stats import t as tdist
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; IDS = ("e61", "u7y", "ngh"); SEEDS = range(5); BOUND = 6 / 7
LAGS = {"R": np.array([2, 3, 4, 5, 6, 7]), "Rthin": np.array([3, 5, 7, 9, 11, 13])}
out = {}


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return M.block_scores(D, st, blk), M.complete_blocks(mon, st, blk)


def exceed(S, lags, j_from):
    """exceedance indicators for test blocks j >= j_from (window must exist)."""
    n = len(S); js = np.arange(max(j_from, lags.max()), n)
    if len(js) == 0: return js, np.zeros(0, bool)
    W = S[js[:, None] - lags[None, :]].max(1); return js, S[js] > W


SEQ = {(sid, s): None for sid, s in itertools.product(IDS, SEEDS)}
for sid, s in SEQ:
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); S, nc = stream(z["st"], z["sq"], z["hp3"])
    SEQ[(sid, s)] = [S[k][:nc[k]].astype(float) for k in range(3)]

# ---------------------------------------------------------------- B1 paired on the R-thin test set
b1 = {}
for sid in IDS:
    rr = {"R": [], "Rthin": []}
    for s in SEEDS:
        NT = {"R": [0, 0], "Rthin": [0, 0]}
        for S in SEQ[(sid, s)]:
            for w in LAGS:
                js, e = exceed(S, LAGS[w], 13); NT[w][0] += e.sum(); NT[w][1] += len(js)
        for w in LAGS: rr[w].append(6 * NT[w][0] / NT[w][1] if NT[w][1] else np.nan)
    b1[sid] = dict(R=float(np.nanmean(rr["R"])), Rthin=float(np.nanmean(rr["Rthin"])), diff=float(np.nanmean(np.array(rr["R"]) - np.array(rr["Rthin"]))),
                   test_h=float(np.mean([sum(max(len(S) - 13, 0) for S in SEQ[(sid, s)]) for s in SEEDS]) / 6))
d = np.array([b1[s]["diff"] for s in IDS])
b1["mean_of_persons"] = dict(R=float(np.mean([b1[s]["R"] for s in IDS])), Rthin=float(np.mean([b1[s]["Rthin"] for s in IDS])), diff=float(d.mean()),
                             ci90_t=[float(d.mean() - tdist.ppf(0.95, 2) * d.std(ddof=1) / np.sqrt(3)), float(d.mean() + tdist.ppf(0.95, 2) * d.std(ddof=1) / np.sqrt(3))])
out["B1_paired"] = b1; print("B1", json.dumps(b1))

# ---------------------------------------------------------------- B2 lag-1 / lag-2 autocorrelation
def acf(x, lag):
    x = x - x.mean(); return float(np.sum(x[lag:] * x[:-lag]) / np.sum(x * x))


b2 = {}
for detr in (False, True):
    per_person = {}
    for sid in IDS:
        res = {}
        for lag in (1, 2):
            zs, ns = [], []
            for k in range(3):
                n = len(SEQ[(sid, 0)][k])
                if n < 8: continue
                zz = []
                for s in SEEDS:
                    S = SEQ[(sid, s)][k]; x = S - np.polyval(np.polyfit(np.arange(len(S)), S, 1), np.arange(len(S))) if detr else S
                    zz.append(np.arctanh(np.clip(acf(x, lag), -0.999, 0.999)))
                zs.append(np.mean(zz)); ns.append(n - 3)
            zs, ns = np.array(zs), np.array(ns)
            zp = float(np.sum(zs * ns) / ns.sum()); se = 1 / np.sqrt(ns.sum())
            res[f"lag{lag}"] = dict(r=float(np.tanh(zp)), ci90=[float(np.tanh(zp - 1.645 * se)), float(np.tanh(zp + 1.645 * se))], n_eff=int(ns.sum()),
                                    _z=zp, _w=float(ns.sum()))
        per_person[sid] = res
    pooled = {}
    for lag in (1, 2):
        zp = sum(per_person[s][f"lag{lag}"]["_z"] * per_person[s][f"lag{lag}"]["_w"] for s in IDS) / sum(per_person[s][f"lag{lag}"]["_w"] for s in IDS)
        se = 1 / np.sqrt(sum(per_person[s][f"lag{lag}"]["_w"] for s in IDS))
        pooled[f"lag{lag}"] = dict(r=float(np.tanh(zp)), ci90=[float(np.tanh(zp - 1.645 * se)), float(np.tanh(zp + 1.645 * se))])
    b2["detrended" if detr else "raw"] = dict(per_person=per_person, pooled=pooled)
out["B2_acf"] = b2; print("B2", json.dumps({k: v["pooled"] for k, v in b2.items()}))
for k, v in b2.items():
    for sid in IDS: print("   ", k, sid, {l: (round(x["r"], 3), [round(c, 3) for c in x["ci90"]]) for l, x in v["per_person"][sid].items()})

# ---------------------------------------------------------------- B3 power of the H-R t test
PROF = [[len(SEQ[(sid, 0)][k]) for k in range(3)] for sid in IDS]


def gen_cell(g, n, model, par):
    if model == "iid": return g.standard_normal(n)
    if model == "ma1":
        e = g.standard_normal(n + 1); return (e[1:] + par * e[:-1]) / np.sqrt(1 + par ** 2)
    if model == "drift": return g.standard_normal(n) + par * np.arange(n)


def rep(g, model, par, frac):
    xr = {"R": [], "Rthin": []}; dif = []
    for i in range(13):
        prof = PROF[g.integers(0, 3)]; N = {w: [0, 0] for w in LAGS}; Np = {w: [0, 0] for w in LAGS}
        for n in prof:
            m = model if (frac >= 1 or g.random() < frac) else "iid"
            S = gen_cell(g, n, m, par)
            for w in LAGS:
                js, e = exceed(S, LAGS[w], 7 if w == "R" else 13); N[w][0] += e.sum(); N[w][1] += len(js)
                js2, e2 = exceed(S, LAGS[w], 13); Np[w][0] += e2.sum(); Np[w][1] += len(js2)
        for w in LAGS: xr[w].append(6 * N[w][0] / N[w][1])
        if Np["R"][1]: dif.append(6 * Np["R"][0] / Np["R"][1] - 6 * Np["Rthin"][0] / Np["Rthin"][1])
    rej = {}
    for w in LAGS:
        x = np.array(xr[w]); sd = x.std(ddof=1); tstat = (x.mean() - BOUND) / (sd / np.sqrt(13)) if sd > 0 else 0
        rej[w] = tstat > tdist.ppf(0.95, 12)
    return rej, float(np.mean(dif)), {w: float(np.mean(xr[w])) for w in LAGS}


b3 = {}
for name, model, par, frac in [("iid", "iid", 0, 1), ("MA1_0.5", "ma1", 0.5, 1), ("MA1_1.0", "ma1", 1.0, 1),
                               ("drift_all", "drift", 0.14, 1), ("drift_half", "drift", 0.14, 0.5), ("drift_all_half_slope", "drift", 0.07, 1)]:
    pr = {w: [] for w in LAGS}; df_ = []; mr = {w: [] for w in LAGS}
    for seed in SEEDS:
        g = np.random.default_rng(900 + seed); r = [rep(g, model, par, frac) for _ in range(4000)]
        for w in LAGS:
            pr[w].append(np.mean([a[w] for a, _, _ in r])); mr[w].append(np.mean([c[w] for _, _, c in r]))
        df_.append(np.mean([b for _, b, _ in r]))
    b3[name] = dict(power={w: float(np.mean(v)) for w, v in pr.items()}, power_sd={w: float(np.std(v)) for w, v in pr.items()},
                    mean_rate={w: float(np.mean(v)) for w, v in mr.items()}, mean_paired_diff_R_minus_Rthin=float(np.mean(df_)))
    print("B3", name, json.dumps(b3[name]), flush=True)
out["B3_power"] = dict(profiles=PROF, beta_z_source="ngh vigorous: slope 2.1-3.5 bpm/block, residual SD 18-21 bpm -> 0.10-0.19 SD/block, mean 0.14",
                       results=b3)
json.dump(out, open(OUT / "b_rchecks.json", "w"), indent=1)
