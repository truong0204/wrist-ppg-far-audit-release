"""S2 r3 (Cowork proof check, PI prompt S1f). DEV ONLY (WildPPG e61/u7y/ngh, sigma 3, E2 seeds 0-4) + simulations.
P1  HR grid step of the HMM output; per K cell: fraction of test blocks with S == tau (ties), fraction of blocks with
    S = -inf (no 3 consecutive windows).
P3  MA(1) Gaussian check of Remark 3 (Cowork table): P(S_j > max window) for the R window (lags 2..7) and the R-thin
    window (lags 3, 5, ..., 13), iid / MA(1) theta 0.5 / 1.0; 5 seeds x 400,000.
    R-thin on dev: FAR per person (seed-averaged), person-cluster bootstrap, sensitivity (clean-stream tau), masking.
Q2  size of the H-R test under iid blocks: 13 persons, block profiles drawn from the 3 dev persons (seed 0), estimand =
    unweighted mean over persons of the person rate 6 sum N / sum T; one-sided one-sample t vs 6/7 at 0.05 (primary),
    Wilcoxon signed-rank and person-cluster bootstrap (sensitivity); bootstrap-t if size(t) > 0.06.
    5 seeds x 20,000 replicates (bootstrap: 5 x 2,000, B = 999).
    uv run python 05_experiments/s2/r3_checks.py
"""
import sys, json, pathlib, itertools, numpy as np
from scipy.stats import t as tdist, wilcoxon, ttest_1samp
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; IDS = ("e61", "u7y", "ngh"); SEEDS = range(5); WIN = 270.0; BOUND = 6 / 7
LAGS = {"R": [2, 3, 4, 5, 6, 7], "Rthin": [3, 5, 7, 9, 11, 13]}
out = {}


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return dict(mon=mon, blk=blk, D=D, S=M.block_scores(D, st, blk), nc=M.complete_blocks(mon, st, blk))


def tau_win(S, nc, lags):
    j0 = max(lags); return {j: float(np.max(S[[j - l for l in lags]])) for j in range(j0, nc)}


# ---------------------------------------------------------------- P1 grid step and ties
z = np.load(C / "WildPPG_e61_seed0.npz"); u = np.unique(z["hp3"][np.isfinite(z["hp3"])]); steps = np.diff(u)
out["P1_grid"] = dict(n_distinct=int(len(u)), step_min=float(steps.min()), step_median=float(np.median(steps)),
                      step_max=float(steps.max()))
ties = []
for sid, s in itertools.product(IDS, SEEDS):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
    for k in range(3):
        nc = cl["nc"][k]; S = cl["S"][k][:nc]
        if nc < 7: continue
        tau = S[:6].max(); Tt = S[6:]
        ties.append(dict(sid=sid, seed=s, state=["rest", "light", "vigorous"][k], T=int(len(Tt)),
                         frac_tie=float(np.mean(Tt == tau)), frac_exceed=float(np.mean(Tt > tau)),
                         frac_neginf=float(np.mean(~np.isfinite(S)))))
cells = {}
for r in ties: cells.setdefault((r["sid"], r["state"]), []).append(r)
out["P1_ties"] = [dict(sid=a, state=b, T=v[0]["T"], frac_tie=float(np.mean([x["frac_tie"] for x in v])),
                       frac_exceed=float(np.mean([x["frac_exceed"] for x in v])), frac_neginf=float(np.mean([x["frac_neginf"] for x in v])))
                  for (a, b), v in cells.items()]
out["P1_ties_pooled"] = dict(frac_tie=float(np.mean([r["frac_tie"] for r in ties])),
                             frac_neginf=float(np.mean([r["frac_neginf"] for r in ties])))
print("P1 grid", out["P1_grid"]); print("P1 ties", out["P1_ties_pooled"])
for c in out["P1_ties"]: print("   ", c)

# ---------------------------------------------------------------- P3 MA(1) simulation
sim = {}
for name, theta in (("iid", 0.0), ("MA1_0.5", 0.5), ("MA1_1.0", 1.0)):
    v = {w: [] for w in LAGS}
    for seed in SEEDS:
        g = np.random.default_rng(seed); e = g.standard_normal((400_000, 15)); X = e[:, 1:] + theta * e[:, :-1]   # X[:, 0..13]
        j = 13
        for w, lags in LAGS.items():
            v[w].append(float(np.mean(X[:, j] > X[:, [j - l for l in lags]].max(1))))
    sim[name] = {w: dict(mean=float(np.mean(x)), sd=float(np.std(x)), rate_per_h=6 * float(np.mean(x))) for w, x in v.items()}
out["P3_MA1"] = sim; print("P3 MA(1)", json.dumps(sim))

# ---------------------------------------------------------------- R-thin (and R) on dev
rows, sens = [], {w: {d: [] for d in (20, 30, 40)} for w in LAGS}
pools = {}
for sid, s in itertools.product(IDS, SEEDS):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
    tauK = M.main_thresholds(cl["S"], cl["nc"]); per = {w: {} for w in LAGS}
    for k in range(3):
        nc = cl["nc"][k]; S = cl["S"][k][:nc]
        for w, lags in LAGS.items():
            th = tau_win(S, nc, lags)
            per[w].update({(k, j): t for j, t in th.items()})
            rows.append(dict(sid=sid, seed=s, state=k, rule=w, T=len(th), N=int(sum(S[j] > t for j, t in th.items()))))
    ce = {k: (2.0 * np.flatnonzero((st == k) & (cl["blk"] == 5))[-1] + 4 if tauK[k] is not None else np.inf) for k in range(3)}
    kof = lambda t: int(st[min(int((t - 4) // 2), len(st) - 1)])
    for d in (20, 30, 40):
        inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"])
        ons = [t for t in z[f"on{d}"] if tauK[kof(t)] is not None and t > ce[kof(t)]]
        for w in LAGS:
            thr = np.array([per[w].get((int(st[i]), int(inj["blk"][i])), np.inf) for i in range(len(st))])
            a = M.alarms(inj["D"], st, inj["blk"], thr, np.isfinite(thr)); c = 2.0 * a + 4
            sens[w][d] += [bool(np.any((c >= t) & (c <= t + WIN))) for t in ons]
        c = 2.0 * np.arange(len(st)) + 4
        for t in z[f"on{d}"]:
            k = kof(t); e = inj["D"][(c >= t) & (c <= t + WIN) & (st == k)]; e = e[np.isfinite(e)]
            if len(e): pools.setdefault((sid, k, s, d), []).append(float(e.max()))


def person_rates(rule):
    pr = {}
    for sid in IDS:
        N = np.mean([sum(r["N"] for r in rows if r["sid"] == sid and r["seed"] == s and r["rule"] == rule) for s in SEEDS])
        T = np.mean([sum(r["T"] for r in rows if r["sid"] == sid and r["seed"] == s and r["rule"] == rule) for s in SEEDS])
        pr[sid] = dict(rate=float(6 * N / T) if T else None, test_h=float(T / 6))
    return pr


g = np.random.default_rng(7); bi = g.integers(0, 3, (10000, 3))
dev = {}
for w in LAGS:
    pr = person_rates(w); x = np.array([pr[i]["rate"] for i in IDS])
    tt = ttest_1samp(x, BOUND, alternative="greater"); half = tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    dev[w] = dict(per_person=pr, mean_of_persons=float(x.mean()), t_p_one_sided=float(tt.pvalue),
                  ci90_two_sided=[float(x.mean() - half), float(x.mean() + half)],
                  boot_q05_mean=float(np.quantile(x[bi].mean(1), 0.05)),
                  sens={d: float(np.mean(v)) for d, v in sens[w].items()}, n_events={d: len(v) for d, v in sens[w].items()},
                  cells_tested=int(len({(r["sid"], r["state"]) for r in rows if r["rule"] == w and r["T"] > 0})))
out["dev_R_vs_Rthin"] = dev; print("dev R vs R-thin", json.dumps(dev))

# masking (block-level, as controls.py), R vs R-thin, Delta 30
mask = {w: {h: [0, 0, 0] for h in range(1, 15)} for w in LAGS}
for sid, s in itertools.product(IDS, SEEDS):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
    for k in range(3):
        nc = cl["nc"][k]; S = cl["S"][k][:nc]; pool = np.array(pools.get((sid, k, s, 30), []))
        if nc < 13 + 15 or len(pool) < 3: continue
        base = {w: tau_win(S, nc, l) for w, l in LAGS.items()}
        for j0 in range(13, nc - 14):
            for _ in range(10):
                e1, e2 = g.choice(pool), g.choice(pool); S1 = S.copy(); S1[j0] = max(S[j0], e1)
                for w, l in LAGS.items():
                    alt = tau_win(S1, nc, l)
                    for h in range(1, 15):
                        rec = mask[w][h]; rec[0] += e2 > base[w][j0 + h]; rec[1] += e2 > alt[j0 + h]; rec[2] += 1
out["masking_d30"] = {w: {h: dict(no_prior=v[0] / v[2], after=v[1] / v[2], n=v[2]) for h, v in m.items() if v[2]} for w, m in mask.items()}
for w in LAGS:
    print("masking", w, [(h, round(v["no_prior"], 2), round(v["after"], 2)) for h, v in out["masking_d30"][w].items()])

# ---------------------------------------------------------------- Q2 size of the H-R tests under iid
PROF = []
for sid in IDS:
    z = np.load(C / f"WildPPG_{sid}_seed0.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"]); PROF.append([cl["nc"][k] for k in range(3)])


def person_rate_iid(gen, nc_list, lags):
    N = T = 0
    for nc in nc_list:
        S = gen.random(nc); th = tau_win(S, nc, lags)
        N += sum(S[j] > t for j, t in th.items()); T += len(th)
    return 6 * N / T


size = {}
for w, lags in LAGS.items():
    res = {"t": [], "wilcoxon": [], "boot_percentile": [], "boot_t": []}
    for seed in SEEDS:
        gen = np.random.default_rng(100 + seed); rej = dict(t=0, wilcoxon=0, boot_percentile=0); rbt = 0
        R_ = 20000
        for rep in range(R_):
            x = np.array([person_rate_iid(gen, PROF[gen.integers(0, 3)], lags) for _ in range(13)])
            m, sd = x.mean(), x.std(ddof=1); tstat = (m - BOUND) / (sd / np.sqrt(13)) if sd > 0 else 0.0
            rej["t"] += tstat > tdist.ppf(0.95, 12)
            d = x - BOUND
            if np.any(d != 0): rej["wilcoxon"] += wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue < 0.05
            bb = x[gen.integers(0, 13, (999, 13))]; rej["boot_percentile"] += np.quantile(bb.mean(1), 0.05) > BOUND
            if rep < 2000:     # bootstrap-t (studentized), cheaper subset
                bm, bs = bb.mean(1), bb.std(1, ddof=1); ok = bs > 0
                tb = (bm[ok] - m) / (bs[ok] / np.sqrt(13))
                rbt += tstat > np.quantile(tb, 0.95)
        for k_ in rej: res[k_].append(rej[k_] / R_)
        res["boot_t"].append(rbt / 2000)
        print("size", w, seed, {k_: round(v[-1], 4) for k_, v in res.items()}, flush=True)
    size[w] = {k_: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k_, v in res.items()}
out["Q2_size"] = dict(profiles_nc=PROF, results=size); print("size", json.dumps(size))
json.dump(out, open(OUT / "r3_checks.json", "w"), indent=1)
