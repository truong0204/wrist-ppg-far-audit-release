"""S1e (Cowork r2 S1, S2), DEV ONLY. WildPPG e61/u7y/ngh, sigma 3, E2 seeds 0-4.

S1  Power of p_order (primary attribution test, prereg rev 4 §4.2) with 13 persons, by simulation from dev blocks.
    Synthetic person = one dev person's 3 cells (complete-block sequences of a random E2 seed), each sequence
    randomly PERMUTED (removes the dev time order -> exchangeable). Alternatives add a linear drift
    beta * j (bpm per block, j = block index) to a fraction f of cells, or to all cells ("common", L1).
    beta from ngh vigorous (linear slope 2.1-3.5 bpm/block over seeds, mean 2.8): c x 2.8 with c in {0.5, 1}.
    p_order = (1 + #{b: X_oracle,b >= X_K}) / (B + 1), B = 999 (oracle = 6 random calibration blocks per cell);
    reject if p_order < 0.05. X2 (H-K) is computed on the same data for comparison (500 bootstrap draws).
    One sequence per cell (no seed averaging in the simulation). 5 seeds x 1,000 replicates per scenario.
S2  Exogenous covariates on dev. Altitude and wrist temperature from the authors' HF file (variables
    data_altitude_values, data_temp_wrist: one value per 8 s; no PPG is loaded; 0 = missing). Per K test block:
    dAlt = |mean altitude of the block - mean altitude of the 6 calibration blocks| (km), dTemp likewise (deg C).
    Block-level logistic model exceed ~ dAlt + dTemp, stacked over cells and persons, per E2 seed; coefficients
    averaged over seeds; person-cluster bootstrap 90 % CI (3 persons: coarse). Descriptive: exceedance rate by
    dAlt tercile.
    uv run python 05_experiments/s2/s1e_attrib.py
"""
import os, sys, json, pathlib, itertools, numpy as np, scipy.io as sio
from scipy.optimize import minimize
from scipy.stats import betabinom
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M, io_wildppg as W
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; IDS = ("e61", "u7y", "ngh"); SEEDS = range(5); R_ = 2.0; NC = 6
out = {}


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return dict(mon=mon, blk=blk, D=D, S=M.block_scores(D, st, blk), nc=M.complete_blocks(mon, st, blk))


# ---------------------------------------------------------------- dev block sequences
SEQ = {}            # (sid, seed) -> [S_k for k in 0..2] (complete blocks only)
BLK = {}
for sid, s in itertools.product(IDS, SEEDS):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
    SEQ[(sid, s)] = [cl["S"][k][:cl["nc"][k]].astype(float) for k in range(3)]; BLK[(sid, s)] = (st, cl)


# ---------------------------------------------------------------- S1 power of p_order
def p0_exceed(T):
    n = np.arange(T + 1); return float(betabinom.pmf(n, T, 1, 6)[6 * n / T > R_].sum())


def one_rep(g, f, beta, common, B=999):
    cells = []
    for i in range(13):
        sid = IDS[g.integers(0, 3)]; s = int(g.integers(0, 5))
        for k in range(3):
            S = SEQ[(sid, s)][k]
            if len(S) < NC + 1: continue
            S = g.permutation(S)
            if common or g.random() < f:
                S = S + beta * np.arange(len(S))
            cells.append((i, S))
    XK = 0; Xo = np.zeros(B); d = np.zeros(13)
    for i, S in cells:
        n = len(S); T = n - NC
        ex = (6 * np.sum(S[NC:] > S[:NC].max()) / T) > R_; XK += ex
        keys = g.random((B, n)); cal = np.argpartition(keys, NC - 1, axis=1)[:, :NC]
        cm = S[cal].max(1); No = (S[None, :] > cm[:, None]).sum(1); Xo += (6 * No / T) > R_
        d[i] += ex - p0_exceed(T)
    p = (1 + np.sum(Xo >= XK)) / (B + 1)
    bi = g.integers(0, 13, (500, 13)); x2 = np.quantile(d[bi].sum(1), 0.05) > 0
    return p < 0.05, x2, XK


SC = [("H0", 0.0, 0.0, False), ("f0.2_c0.5", 0.2, 1.4, False), ("f0.2_c1", 0.2, 2.8, False),
      ("f0.4_c0.5", 0.4, 1.4, False), ("f0.4_c1", 0.4, 2.8, False), ("common_c0.5", 1.0, 1.4, True), ("common_c1", 1.0, 2.8, True)]
pw = {}
if os.environ.get("SKIP_S1"):          # rerun S2 only; keep the stored S1 results
    pw = json.load(open(OUT / "s1e_attrib.json"))["S1_power_p_order"]; SC = []
for name, f, beta, common in SC:
    po, x2, xk = [], [], []
    for seed in SEEDS:
        g = np.random.default_rng(500 + seed); r = [one_rep(g, f, beta, common) for _ in range(1000)]
        po.append(np.mean([a for a, _, _ in r])); x2.append(np.mean([b for _, b, _ in r])); xk.append(np.mean([c for _, _, c in r]))
    pw[name] = dict(frac=f, beta_bpm_per_block=beta, common=common, power_p_order=float(np.mean(po)), sd_p_order=float(np.std(po)),
                    power_X2=float(np.mean(x2)), mean_XK=float(np.mean(xk)))
    print("S1", name, json.dumps(pw[name]), flush=True)
out["S1_power_p_order"] = pw

# ---------------------------------------------------------------- S2 exogenous covariates (dev)
hf = sio.loadmat(W.HF, variable_names=["data_altitude_values", "data_temp_wrist"])
mp = json.load(open(ROOT / "05_experiments/s1b/results/B_wildppg.json"))["mapping"]
COV = {}
for sid in IDS:
    i = mp[sid][0]
    a = np.asarray(hf["data_altitude_values"][i, 0], float).ravel(); t = np.asarray(hf["data_temp_wrist"][i, 0], float).ravel()
    a[a == 0] = np.nan; t[t == 0] = np.nan; COV[sid] = (a / 1000.0, t)     # km, deg C


def block_cov(sid, st, blk, k, js):
    a, t = COV[sid]; idx_hf = np.minimum((2 * np.arange(len(st)) + 4) // 8, len(a) - 1).astype(int)
    res = []
    for j in js:
        w = (st == k) & (blk == j)
        res.append((np.nanmean(a[idx_hf[w]]), np.nanmean(t[idx_hf[w]]), np.mean(2.0 * np.flatnonzero(w) + 4) / 3600))   # alt km, temp C, time h
    return np.array(res)


COEF_MAX = 20.0      # rev 6 rule: |coefficient| > 20 is treated as (quasi-)complete separation


def logit_fit(X, y, strict=False):
    X1 = np.column_stack([np.ones(len(y)), X])
    nll = lambda b: np.sum(np.logaddexp(0, X1 @ b) - y * (X1 @ b)) + 1e-4 * np.sum(b[1:] ** 2)
    r = minimize(nll, np.zeros(X1.shape[1]), method="BFGS")
    if strict and (not r.success or not np.all(np.isfinite(r.x)) or np.max(np.abs(r.x[1:])) > COEF_MAX):
        return None
    return r.x


rows = {s: [] for s in SEEDS}
for sid, s in itertools.product(IDS, SEEDS):
    st, cl = BLK[(sid, s)]
    for k in range(3):
        nc = cl["nc"][k]
        if nc < NC + 1: continue
        S = cl["S"][k][:nc]; tau = S[:NC].max()
        cal = block_cov(sid, st, cl["blk"], k, range(NC)).mean(0); te = block_cov(sid, st, cl["blk"], k, range(NC, nc))
        for jj, (a_, t_, h_) in enumerate(te):
            rows[s].append(dict(sid=sid, state=k, j=NC + jj, exceed=int(S[NC + jj] > tau), dalt=abs(a_ - cal[0]), dtemp=abs(t_ - cal[1]),
                                dtime=h_ - cal[2]))      # rev 5: real time elapsed since calibration (h; block mean - calibration mean)
DROPPED = {}


def fit_model(cols, seed0=11):
    """rev 6 rule (prereg §4.2 d): a bootstrap draw is dropped if the outcome has no variation, the optimiser fails,
    or the fit separates (|coef| > COEF_MAX); the number dropped is reported; if > 5 % of draws are dropped, only the
    tercile description is reported for that model."""
    coef, boot = [], []; g = np.random.default_rng(seed0); n_draw = n_drop = 0
    for s in SEEDS:
        R = rows[s]; X = np.array([[r[c] for c in cols] for r in R]); y = np.array([r["exceed"] for r in R], float)
        coef.append(logit_fit(X, y))
        for _ in range(300):
            n_draw += 1
            ids = g.choice(IDS, 3); RR = [r for i in ids for r in R if r["sid"] == i]
            Xb = np.array([[r[c] for c in cols] for r in RR]); yb = np.array([r["exceed"] for r in RR], float)
            fit = None if yb.min() == yb.max() else logit_fit(Xb, yb, strict=True)
            if fit is None:
                n_drop += 1; continue
            boot.append(fit)
    DROPPED["+".join(cols)] = dict(draws=n_draw, dropped=n_drop, frac=n_drop / n_draw, descriptive_only=n_drop / n_draw > 0.05)
    return np.mean(coef, 0), np.array(boot)


# rev 4: primary model = dAlt alone; secondary = dTemp alone; joint model descriptive (collinear on this route)
single = {}
for c in ("dalt", "dtemp"):
    cf, bt = fit_model([c]); single[c] = dict(coef=float(cf[1]), ci90=[float(np.percentile(bt[:, 1], 5)), float(np.percentile(bt[:, 1], 95))])
allrows = [r for s in SEEDS for r in rows[s]]
corr_alt_temp = float(np.corrcoef([r["dalt"] for r in allrows], [r["dtemp"] for r in allrows])[0, 1])
coef, boot = fit_model(["dalt", "dtemp"])
# rev 5 (PI): sensitivity model with elapsed real time; correlation of dAlt with time
ci = lambda b, i: [float(np.percentile(b[:, i], 5)), float(np.percentile(b[:, i], 95))]
cf_t, bt_t = fit_model(["dalt", "dtime"]); cf_to, bt_to = fit_model(["dtime"])
corr_alt_time = float(np.corrcoef([r["dalt"] for r in allrows], [r["dtime"] for r in allrows])[0, 1])
rev5 = dict(sens_dalt_plus_time=dict(dalt=float(cf_t[1]), dalt_ci90=ci(bt_t, 1), dtime_per_h=float(cf_t[2]), dtime_ci90=ci(bt_t, 2)),
            sens_time_only=dict(dtime_per_h=float(cf_to[1]), ci90=ci(bt_to, 1)),
            sens_joint_alt_temp=dict(dalt=float(coef[1]), dalt_ci90=ci(boot, 1), dtemp=float(coef[2]), dtemp_ci90=ci(boot, 2)),
            corr_dalt_dtime=corr_alt_time, corr_dalt_dtemp=corr_alt_temp,
            dtime_range_h=[float(min(r["dtime"] for r in allrows)), float(max(r["dtime"] for r in allrows))])
rev5["bootstrap_dropped"] = DROPPED
print("rev5", json.dumps(rev5))
allr = [r for s in SEEDS for r in rows[s]]; da = np.array([r["dalt"] for r in allr]); ex = np.array([r["exceed"] for r in allr])
q = np.quantile(da, [1 / 3, 2 / 3])
terc = [float(ex[da <= q[0]].mean()), float(ex[(da > q[0]) & (da <= q[1])].mean()), float(ex[da > q[1]].mean())]
out["S2_covariates"] = dict(
    n_test_blocks_per_seed=len(rows[0]), coef_intercept=float(coef[0]), coef_dalt_per_km=float(coef[1]), coef_dtemp_per_degC=float(coef[2]),
    ci90_dalt=[float(np.percentile(boot[:, 1], 5)), float(np.percentile(boot[:, 1], 95))],
    ci90_dtemp=[float(np.percentile(boot[:, 2], 5)), float(np.percentile(boot[:, 2], 95))],
    dalt_tercile_edges_km=[float(x) for x in q], exceed_rate_by_dalt_tercile=terc,
    dalt_range_km=[float(da.min()), float(da.max())],
    primary_dalt_only=single["dalt"], secondary_dtemp_only=single["dtemp"], corr_dalt_dtemp=corr_alt_temp, rev5=rev5,
    note="altitude and time are confounded on this route; 3-person cluster bootstrap is coarse")
print("S2", json.dumps(out["S2_covariates"]))
json.dump(out, open(OUT / "s1e_attrib.json", "w"), indent=1)
