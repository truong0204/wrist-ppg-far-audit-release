"""Confirmatory pipeline of prereg v2.2 (rev 8). ONE code path for the dev dry run and for the confirmatory run.

Every function below is MOVED from a dev script of 05_experiments/s2 (source named in each docstring), with the
logic unchanged; only the packaging (functions, explicit seeds, explicit ID list) is new. The dry run on the 3 dev IDs
is regression-checked against the numbers recorded in the prereg (§5) by 05_experiments/confirm/regression_check.py.

All random seeds are fixed constants in this file. Seeds of the E2 estimator: 0-4. HMM sigma = 3 only.

Missing data (prereg rev 8, §4.0, PI 23-11):
- sum-type quantities (cells above r, d_i of X2 / Y2 / H-K1): a seed without test blocks contributes 0, divided by 5;
- rate-type quantities (person rates of K, R, R-thin, R - R-thin, sensitivity): mean over the seeds with >= 1 test
  block of that rule (number of seeds reported), never 0 for an undefined rate;
- a person is excluded from a test only if it has no test block of that rule in ALL 5 seeds (H-K keeps everyone with
  d_i = 0); actual n is reported and written into the sentences;
- a raw file that cannot be opened, or lacks a required variable, is read once more; if it fails again the person is
  excluded from every analysis (deviation, logged); its partial data are never loaded.
"""
from __future__ import annotations
import json, os, pathlib, itertools
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from scipy.stats import betabinom, spearmanr, rankdata, ttest_1samp, wilcoxon, t as tdist
from . import io_wildppg as W, features, e2cnn, inject, method as M

ROOT = pathlib.Path(__file__).resolve().parents[2]
R1C = ROOT / "05_experiments/s1c/results"
TH = 0.02896360121667385          # SQI gate (prereg §1)
N_CAL, KSTAR = 6, 6
DELTAS = (20, 30, 40)
WIN_DET = 270.0
SEEDS = (0, 1, 2, 3, 4)
STN = ["rest", "light", "vigorous"]
R_TARGET = 2.0
BOUND_R = 6 / 7                    # expectation bound of R / R-thin (0.857 per monitored hour)
LAGS = {"R": np.array([2, 3, 4, 5, 6, 7]), "Rthin": np.array([3, 5, 7, 9, 11, 13])}
FIRST_TEST = {"K": 6, "Km1": 13, "R": 7, "Rthin": 13, "Rsel": 7, "ACI": 6}
ACI_GAMMA, ACI_ALPHA = 0.05, 1 / 7
# fixed seeds (rev 7)
SEED_BOOT = 0          # person-cluster bootstraps of H-K (X2, Y2), H-K1, FAR, H-R sensitivity, matched-FAR, association
N_BOOT = 10000
N_ORACLE = 2000        # p_order draws
SEED_COV = 11          # covariate pairs bootstrap
N_COV_PER_SEED = 400   # 5 seeds x 400 = 2,000 draws (prereg §4.2 d)
COEF_MAX = 20.0
SEED_MASK = 3
SEED_ASSOC = 20261123


# =========================================================================== part 1: E2 inference (precompute.py)
def _run(m, ppg, acc, n):
    """precompute.py run(), sigma 3 only."""
    P = e2cnn.predict_proba(m, features.spectra(ppg, acc, n))
    return P.max(1).astype(np.float32), e2cnn.hmm_track_grid(P, e2cnn.FCN_GRID, 3.0).astype(np.float32)


class Unreadable(Exception):
    """raw file could not be opened or lacks a required variable (prereg rev 8 §4.0)."""


def open_raw(sid):
    """W.load_wrist (open the raw file, required variables wrist.ppg_g / acc_x/y/z and the HF reference) with one
    retry. PermissionError (the confirmatory lock) is never caught."""
    errs = []
    for attempt in (1, 2):
        try:
            return W.load_wrist(sid)
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001  any failure to open the file or find a required variable
            errs.append(f"attempt {attempt}: {type(e).__name__}: {e}")
    raise Unreadable("; ".join(errs))


def load_recording(sid):
    """precompute.py recs() for WildPPG. W.load_wrist raises PermissionError for a confirmatory ID while
    io_wildppg.CONFIRMATORY_PPG_OPEN is False."""
    d = open_raw(sid); p, a = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
    n = (len(p) // 64 - 8) // 2 + 1
    ref2 = np.repeat(d["hr_ref"], 4)[:n]; ref2 = np.concatenate([ref2, np.full(n - len(ref2), np.nan)])
    return p, a, n, ref2


def infer(sid, cache_dir: pathlib.Path, log=print):
    """precompute.py main loop for one ID, all seeds; clean, injected (20/30/40) and calibration-contamination
    streams; cached as <cache_dir>/<sid>_seed<k>.npz. Returns None if all arrays exist, else a reason string if the raw
    file is unreadable (rev 8), else True."""
    import torch
    torch.set_num_threads(8)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [k for k in SEEDS if not (cache_dir / f"{sid}_seed{k}.npz").exists()]
    if not todo:
        return None
    try:
        ppg, acc, n, ref2 = load_recording(sid)
    except Unreadable as e:
        log(f"  UNREADABLE {sid} after 2 attempts ({e}); excluded from every analysis (deviation)")
        return str(e)
    st = M.states(acc, n)
    for k in todo:
        m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(R1C / f"fcn_all_seed{k}.pt")); m.eval()
        sq, hp = _run(m, ppg, acc, n); nn = len(sq); st_ = st[:nn]
        save = dict(st=st_, sq=sq, hp3=hp)
        for delta in DELTAS:
            rng = np.random.default_rng(1000 * k + delta + 7)
            on = inject.schedule(nn, np.ones(nn, bool), rng, per_hour=3, min_gap_s=600, dur_s=240, start_s=120)
            pp = ppg.copy(); rr = ref2.copy()
            for o in on:
                pp, rr, _ = inject.inject(pp, 64, rr, o, delta, ramp_s=30, plateau_s=180)
            sqi, hpi = _run(m, pp, acc, n)
            save[f"on{delta}"] = np.array(on); save[f"sq{delta}"] = sqi; save[f"hp3_{delta}"] = hpi
        mon = sq >= TH; blk = M.blocks(mon, st_); nc = M.complete_blocks(mon, st_, blk)
        on_c = []
        for kk in range(3):
            if nc[kk] >= 6:
                i0 = int(np.flatnonzero((st_ == kk) & (blk == 2))[0]); t = 2.0 * i0
                if all(abs(t - o) >= 600 for o in on_c) and t + 240 < 2 * nn:
                    on_c.append(t)
        pp = ppg.copy(); rr = ref2.copy()
        for o in on_c:
            pp, rr, _ = inject.inject(pp, 64, rr, o, 30, ramp_s=30, plateau_s=180)
        sqc, hpc = _run(m, pp, acc, n)
        save["on_cal"] = np.array(on_c); save["sq_cal"] = sqc; save["hp3_cal"] = hpc
        part = cache_dir / f"{sid}_seed{k}.npz.part"   # rev 8: atomic write, so a killed run never leaves a partial array
        with open(part, "wb") as fh:
            np.savez_compressed(fh, **save)
        os.replace(part, cache_dir / f"{sid}_seed{k}.npz")
        log(f"  inferred {sid} seed {k}: {nn} windows, {len(on_c)} calibration events")
    return True


def load_covariates(sid):
    """s1e_attrib.py: altitude (km) and wrist device-internal temperature (C) from the authors' HF file; no PPG."""
    hf = sio.loadmat(W.HF, variable_names=["data_altitude_values", "data_temp_wrist"])
    i = W.hf_index(sid)
    a = np.asarray(hf["data_altitude_values"][i, 0], float).ravel(); t = np.asarray(hf["data_temp_wrist"][i, 0], float).ravel()
    a[a == 0] = np.nan; t[t == 0] = np.nan
    return a / 1000.0, t


# =========================================================================== part 2: shared structures
def stream(st, sq, hp):
    """analysis.py / controls.py stream()."""
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return dict(mon=mon, blk=blk, D=D, S=M.block_scores(D, st, blk), nc=M.complete_blocks(mon, st, blk))


def cal_end_time(st, blk, k):
    """analysis.py."""
    idx = np.flatnonzero((st == k) & (blk == N_CAL - 1))
    return 2.0 * idx[-1] + 4 if len(idx) else np.inf


def detect(alarm_idx, onsets):
    """analysis.py detect(): (detected, delay) per onset."""
    c = 2.0 * alarm_idx + 4; res = []
    for t0 in onsets:
        hit = c[(c >= t0) & (c <= t0 + WIN_DET)]
        res.append((len(hit) > 0, float(hit[0] - t0) if len(hit) else np.nan))
    return res


def onset_state(st, t):
    return int(st[min(int((t - 4) // 2), len(st) - 1)])


def tau_win(S, nc, lags):
    """r3_checks.py: R (lags 2..7) / R-thin (lags 3, 5, ..., 13)."""
    j0 = max(lags); return {j: float(np.max(S[[j - l for l in lags]])) for j in range(j0, nc)}


def tau_rsel(S, nc):
    """controls.py tau_R(selective=True): window = 6 most recent blocks <= j-2 without an alarm."""
    alarmed = np.zeros(nc, bool); thr = {}
    for j in range(N_CAL + 1, nc):
        cand = [i for i in range(j - 2, -1, -1) if not alarmed[i]][:N_CAL]
        t = float(np.max(S[cand])); thr[j] = t; alarmed[j] = S[j] > t
    return thr


def tau_aci(S, nc, astar=ACI_ALPHA):
    """b4_aci.py aci()."""
    a = astar; thr = {}
    for j in range(N_CAL, nc):
        pool = np.sort(S[:j]); k = int(np.ceil((j + 1) * (1 - a)))
        t = np.inf if k > j else (-np.inf if k <= 0 else float(pool[k - 1]))
        thr[j] = t; a = a + ACI_GAMMA * (astar - (S[j] > t))
    return thr


def tau_K(S, nc):
    return {j: float(np.max(S[:N_CAL])) for j in range(N_CAL, nc)} if nc >= N_CAL + 1 else {}


def tau_Km1(S, nc):
    return {j: float(np.max(S[0:12:2])) for j in range(13, nc)} if nc >= 14 else {}


RULES = {"K": tau_K, "Km1": tau_Km1, "R": lambda S, nc: tau_win(S, nc, LAGS["R"]),
         "Rthin": lambda S, nc: tau_win(S, nc, LAGS["Rthin"]), "Rsel": tau_rsel, "ACI": tau_aci}


def block_rule(S, nc, rule):
    """per test block threshold of a rule; Lemma 1: alarms = #{j: S_j > tau_j}."""
    if nc < FIRST_TEST[rule] + 1:
        return {}
    return RULES[rule](S, nc)


def p0_exceed(T):
    """m3_power.py / controls.py: P0(rate > r) under BetaBinomial(T, 1, 6)."""
    n = np.arange(T + 1); return float(betabinom.pmf(n, T, 1, 6)[6 * n / T > R_TARGET].sum())


class Data:
    """all (ID, seed) streams of one run."""

    def __init__(self, ids, cache_dir=None, arrays=None, cov=None):
        """arrays / cov: prebuilt {(sid, seed): npz dict} and {sid: (alt, temp)} (tests only)."""
        self.ids = list(ids); self.z = {}; self.cl = {}; self.S = {}
        for sid, k in itertools.product(self.ids, SEEDS):
            z = arrays[(sid, k)] if arrays is not None else dict(np.load(cache_dir / f"{sid}_seed{k}.npz")); self.z[(sid, k)] = z
            cl = stream(z["st"], z["sq"], z["hp3"]); self.cl[(sid, k)] = cl
            self.S[(sid, k)] = [cl["S"][kk][:cl["nc"][kk]].astype(float) for kk in range(3)]
        self.cov = cov if cov is not None else {sid: load_covariates(sid) for sid in self.ids}


def boot_q(vals, q, seed=SEED_BOOT, B=N_BOOT, stat=np.sum):
    g = np.random.default_rng(seed); v = np.asarray(vals, float); bi = g.integers(0, len(v), (B, len(v)))
    return float(np.quantile(stat(v[bi], axis=1), q))


# =========================================================================== part 3: analyses
def cells_rule(D: Data, rule):
    """per (sid, state, seed): T, N (exceedances = alarms by Lemma 1), rate."""
    out = []
    for (sid, k), Ss in D.S.items():
        for kk, S in enumerate(Ss):
            th = block_rule(S, len(S), rule)
            if not th: continue
            N = int(sum(S[j] > t for j, t in th.items())); T = len(th)
            out.append(dict(sid=sid, state=STN[kk], seed=k, T=T, N=N, rate=6 * N / T))
    return out


def person_counts(cells, sid, seeds=SEEDS):
    """per-seed (N, T) of one person, summed over its cells."""
    return {k: (sum(c["N"] for c in cells if c["sid"] == sid and c["seed"] == k), sum(c["T"] for c in cells if c["sid"] == sid and c["seed"] == k))
            for k in seeds}


def person_rate(cells, sid, seeds=SEEDS):
    """controls.py person_rates / r3_checks.py: 6 x (seed-mean N) / (seed-mean T), the means taken over the seeds with
    >= 1 test block of the rule (rev 8 rate-type rule; identical to the dev formula when all 5 seeds have blocks).
    Returns (rate or NaN, seed-mean N over 5 seeds, seed-mean test hours over 5 seeds, n seeds used,
    mean test blocks over the seeds used); N and hours over 5 seeds feed the pooled (sum-type) rate."""
    pc = person_counts(cells, sid, seeds); used = [k for k in seeds if pc[k][1] > 0]
    N5 = np.mean([pc[k][0] for k in seeds]); T5 = np.mean([pc[k][1] for k in seeds])
    if not used:
        return np.nan, float(N5), float(T5 / 6), 0, 0.0
    Nu = np.mean([pc[k][0] for k in used]); Tu = np.mean([pc[k][1] for k in used])
    return float(6 * Nu / Tu), float(N5), float(T5 / 6), len(used), float(Tu)


def far_summary(D, rule):
    """controls.py far + cluster_boot (pooled rate, person-cluster 95 % percentile CI)."""
    cells = cells_rule(D, rule); pr = {sid: person_rate(cells, sid) for sid in D.ids}
    Ns = np.array([pr[s][1] for s in D.ids]); Hs = np.array([pr[s][2] for s in D.ids])
    g = np.random.default_rng(SEED_BOOT); bi = g.integers(0, len(D.ids), (N_BOOT, len(D.ids)))
    v = Ns[bi].sum(1) / Hs[bi].sum(1)
    per_cell = {}
    for c in cells: per_cell.setdefault((c["sid"], c["state"]), []).append(c)
    have = [s for s in D.ids if pr[s][3] > 0]
    return dict(pooled=float(Ns.sum() / Hs.sum()) if Hs.sum() else None, cluster_ci95=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if Hs.sum() else None,
                per_person={s: pr[s][0] for s in D.ids}, test_h_per_person={s: pr[s][2] for s in D.ids},
                n_seeds_per_person={s: pr[s][3] for s in D.ids}, test_blocks_per_person={s: pr[s][4] for s in D.ids},
                mean_of_persons=float(np.mean([pr[s][0] for s in have])) if have else None, n_persons=len(have),
                per_cell={f"{a}|{b}": dict(T=float(np.mean([x["T"] for x in v])), rate=float(np.mean([x["rate"] for x in v])), n_seeds=len(v),
                                           seeds_above_r=int(sum(x["rate"] > R_TARGET for x in v))) for (a, b), v in per_cell.items()})


def h_k(D):
    """§4.1: X2 (primary), X1, Y2; controls.py prereg dry run + drift_extra.py X1."""
    cells = cells_rule(D, "K")
    for c in cells: c["P0"] = p0_exceed(c["T"]); c["ex"] = c["rate"] > R_TARGET
    d = {s: float(np.mean([sum(c["ex"] - c["P0"] for c in cells if c["sid"] == s and c["seed"] == k) for k in SEEDS])) for s in D.ids}
    dY = {s: float(np.mean([sum(c["N"] - c["T"] / 7 for c in cells if c["sid"] == s and c["seed"] == k) for k in SEEDS])) for s in D.ids}
    x1 = []
    for k in SEEDS:
        cs = [c for c in cells if c["seed"] == k]; X = int(sum(c["ex"] for c in cs)); dist = np.array([1.0])
        for c in cs: dist = np.convolve(dist, [1 - c["P0"], c["P0"]])
        x1.append(dict(seed=k, X=X, E0=float(sum(c["P0"] for c in cs)), p=float(dist[X:].sum())))
    q05 = boot_q(list(d.values()), 0.05); qY = boot_q(list(dY.values()), 0.05)
    rates = far_summary(D, "K")
    return dict(d_i=d, sum_d=float(sum(d.values())), X2_q05=q05, X2_reject=bool(q05 > 0),
                Xbar=float(np.mean([x["X"] for x in x1])), E0bar=float(np.mean([x["E0"] for x in x1])), n_cells=len({(c["sid"], c["state"]) for c in cells}),
                X1=x1, X1_median_p=float(np.median([x["p"] for x in x1])),
                Y2_d_i=dY, Y2_sum=float(sum(dY.values())), Y2_q05=qY, Y2_reject=bool(qY > 0),
                mean_rate_of_persons=rates["mean_of_persons"], n=len(D.ids), n_persons_with_cells=rates["n_persons"], rates=rates)


def h_k1(D):
    """§4.1b (r6_hk1.py): sum over cells of N_c - T_c/7 for K m = 1; person-cluster bootstrap."""
    cells = cells_rule(D, "Km1")
    ids = [s for s in D.ids if any(c["sid"] == s for c in cells)]          # rev 8: excluded only if no block in all 5 seeds
    d = {s: float(np.mean([sum(c["N"] - c["T"] / 7 for c in cells if c["sid"] == s and c["seed"] == k) for k in SEEDS])) for s in ids}
    q05 = boot_q(list(d.values()), 0.05) if ids else None
    k0 = cells_rule_on(D, "K", 13)
    paired = {s: dict(K_m0=person_rate(k0, s)[0], K_m1=person_rate(cells, s)[0]) for s in ids}
    return dict(d_i=d, sum_d=float(sum(d.values())), q05=q05, reject=bool(q05 is not None and q05 > 0), n=len(ids),
                excluded_no_blocks=[s for s in D.ids if s not in ids], paired_same_blocks=paired, rates=far_summary(D, "Km1"))


def cells_rule_on(D, rule, j_from):
    """rule evaluated only on test blocks >= j_from (b_rchecks.py / r6_hk1.py paired comparisons)."""
    out = []
    for (sid, k), Ss in D.S.items():
        for kk, S in enumerate(Ss):
            if len(S) < j_from + 1: continue
            th = {j: t for j, t in block_rule(S, len(S), rule).items() if j >= j_from}
            if not th: continue
            N = int(sum(S[j] > t for j, t in th.items())); out.append(dict(sid=sid, state=STN[kk], seed=k, T=len(th), N=N, rate=6 * N / len(th)))
    return out


def p_order(D):
    """§4.2 a (controls.py, rev 3/5): CRN keys per (person index, state); (1 + #)/(B + 1)."""
    Xk = []; Xo = []
    for k in SEEDS:
        xk = 0; xo = np.zeros(N_ORACLE)
        for pi, sid in enumerate(D.ids):
            for kk, S in enumerate(D.S[(sid, k)]):
                nc = len(S)
                if nc < N_CAL + 1: continue
                T = nc - N_CAL; xk += (6 * np.sum(S[N_CAL:] > S[:N_CAL].max()) / T) > R_TARGET
                keys = np.random.default_rng(1000 * pi + kk).random((N_ORACLE, 200))
                No = np.empty(N_ORACLE, int)
                for b in range(N_ORACLE):
                    cal = np.argsort(keys[b, :nc])[:N_CAL]; msk = np.ones(nc, bool); msk[cal] = False
                    No[b] = np.sum(S[msk] > S[cal].max())
                xo += (6 * No / T) > R_TARGET
        Xk.append(xk); Xo.append(xo)
    Xbar_K = float(np.mean(Xk)); Xbar_o = np.mean(Xo, axis=0)
    return dict(Xbar_K=Xbar_K, Xbar_oracle_mean=float(Xbar_o.mean()), p=float((1 + np.sum(Xbar_o >= Xbar_K - 1e-9)) / (N_ORACLE + 1)))


def _partial_spearman(y, x, Z):
    ry, rx = rankdata(y), rankdata(x); Zr = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]; ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


def descriptive_drift(D):
    """§4.2 b-c (controls.py): rho, partial rho, fragmentation trends; association with person-cluster CI."""
    rows = []
    for (sid, k) in D.z:
        z = D.z[(sid, k)]; st = z["st"]; cl = D.cl[(sid, k)]
        for kk in range(3):
            nc = cl["nc"][kk]
            if nc < N_CAL + 1: continue
            S = cl["S"][kk][:nc]; seg, val = [], []
            for j in range(nc):
                idx = np.flatnonzero((st == kk) & (cl["blk"] == j)); seg.append(1 + int(np.sum(np.diff(idx) > 1)))
                val.append(int(np.isfinite(cl["D"][idx]).sum()))
            j = np.arange(nc); seg, val = np.array(seg), np.array(val)
            rows.append(dict(sid=sid, state=STN[kk], seed=k, rho=float(spearmanr(S, j)[0]), partial_rho=_partial_spearman(S, j, [seg, val]),
                             rho_seg_time=float(spearmanr(seg, j)[0]), rho_valid_time=float(spearmanr(val, j)[0]),
                             rate_K=6 * np.sum(S[N_CAL:] > S[:N_CAL].max()) / (nc - N_CAL)))
    ct = {}
    for r in rows: ct.setdefault((r["sid"], r["state"]), []).append(r)
    tab = [dict(sid=a, state=b, **{key: float(np.nanmean([x[key] for x in v])) for key in ("rho", "partial_rho", "rho_seg_time", "rho_valid_time", "rate_K")})
           for (a, b), v in ct.items()]
    g = np.random.default_rng(SEED_ASSOC); bi = g.integers(0, len(D.ids), (2000, len(D.ids))); assoc = {}
    for key in ("rho", "partial_rho"):
        sp = lambda rr: float(spearmanr([r[key] for r in rr], [r["rate_K"] for r in rr])[0])
        bs = [sp([c for i in b for c in tab if c["sid"] == D.ids[i]]) for b in bi]; bs = [x for x in bs if np.isfinite(x)]
        assoc[key] = dict(spearman=sp(tab), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return dict(cells=tab, association=assoc)


def _logit_fit(X, y, strict=False):
    """s1e_attrib.py logit_fit (rev 6 rule)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    nll = lambda b: np.sum(np.logaddexp(0, X1 @ b) - y * (X1 @ b)) + 1e-4 * np.sum(b[1:] ** 2)
    r = minimize(nll, np.zeros(X1.shape[1]), method="BFGS")
    if strict and (not r.success or not np.all(np.isfinite(r.x)) or np.max(np.abs(r.x[1:])) > COEF_MAX):
        return None
    return r.x


def covariates(D):
    """§4.2 d (s1e_attrib.py, rev 5/6): per K test block dAlt, dTemp, dTime; logistic models; pairs bootstrap."""
    rows = {k: [] for k in SEEDS}
    for (sid, k) in D.z:
        st = D.z[(sid, k)]["st"]; cl = D.cl[(sid, k)]; a, t = D.cov[sid]
        idx_hf = np.minimum((2 * np.arange(len(st)) + 4) // 8, len(a) - 1).astype(int)
        def bc(kk, js):
            res = []
            for j in js:
                w = (st == kk) & (cl["blk"] == j)
                res.append((np.nanmean(a[idx_hf[w]]), np.nanmean(t[idx_hf[w]]), np.mean(2.0 * np.flatnonzero(w) + 4) / 3600))
            return np.array(res)
        for kk in range(3):
            nc = cl["nc"][kk]
            if nc < N_CAL + 1: continue
            S = cl["S"][kk][:nc]; tau = S[:N_CAL].max(); cal = bc(kk, range(N_CAL)).mean(0)
            for jj, (a_, t_, h_) in enumerate(bc(kk, range(N_CAL, nc))):
                rows[k].append(dict(sid=sid, exceed=int(S[N_CAL + jj] > tau), dalt=abs(a_ - cal[0]), dtemp=abs(t_ - cal[1]), dtime=h_ - cal[2]))
    dropped = {}

    def fit(cols):
        g = np.random.default_rng(SEED_COV); coef, boot = [], []; nd = ndr = 0
        for k in SEEDS:
            R = rows[k]; X = np.array([[r[c] for c in cols] for r in R]); y = np.array([r["exceed"] for r in R], float)
            if not R: continue
            coef.append(_logit_fit(X, y)); ids_k = [s for s in D.ids if any(r["sid"] == s for r in R)]   # rev 8: persons with data
            for _ in range(N_COV_PER_SEED):
                nd += 1; ids = g.choice(ids_k, len(ids_k)); RR = [r for i in ids for r in R if r["sid"] == i]
                Xb = np.array([[r[c] for c in cols] for r in RR]); yb = np.array([r["exceed"] for r in RR], float)
                f = None if yb.min() == yb.max() else _logit_fit(Xb, yb, strict=True)
                if f is None: ndr += 1; continue
                boot.append(f)
        dropped["+".join(cols)] = dict(draws=nd, dropped=ndr, frac=ndr / nd, descriptive_only=ndr / nd > 0.05)
        cf = np.mean(coef, 0); bt = np.array(boot)
        return [dict(coef=float(cf[i + 1]), ci90=[float(np.percentile(bt[:, i + 1], 5)), float(np.percentile(bt[:, i + 1], 95))]) for i in range(len(cols))]

    allr = [r for k in SEEDS for r in rows[k]]
    da = np.array([r["dalt"] for r in allr]); ex = np.array([r["exceed"] for r in allr]); q = np.quantile(da, [1 / 3, 2 / 3])
    res = dict(primary_dalt=fit(["dalt"])[0], sens1_dalt_dtemp=dict(zip(["dalt", "dtemp"], fit(["dalt", "dtemp"]))),
               sens2_dalt_dtime=dict(zip(["dalt", "dtime"], fit(["dalt", "dtime"]))), desc_dtemp=fit(["dtemp"])[0], desc_dtime=fit(["dtime"])[0],
               corr_dalt_dtime=float(np.corrcoef(da, [r["dtime"] for r in allr])[0, 1]),
               corr_dalt_dtemp=float(np.corrcoef(da, [r["dtemp"] for r in allr])[0, 1]),
               tercile_edges_km=[float(x) for x in q],
               exceed_by_dalt_tercile=[float(ex[da <= q[0]].mean()), float(ex[(da > q[0]) & (da <= q[1])].mean()), float(ex[da > q[1]].mean())],
               n_test_blocks_per_seed=len(rows[0]),
               n_persons_per_seed={str(k): len({r["sid"] for r in rows[k]}) for k in SEEDS})
    res["bootstrap_dropped"] = dropped
    return res


def h_r(D):
    """§4.3 (r3_checks.py, b_rchecks.py): t test on the unweighted mean of person rates; Wilcoxon, percentile
    bootstrap; R-thin; paired R vs R-thin on blocks >= 13; lag-1/lag-2 autocorrelation."""
    out = {}
    for rule in ("R", "Rthin"):
        cells = cells_rule(D, rule); pr = {s: person_rate(cells, s) for s in D.ids}
        ids = [s for s in D.ids if pr[s][3] > 0]                              # rev 8: excluded only if no block in all 5 seeds
        x = np.array([pr[s][0] for s in ids]); Ns = sum(pr[s][1] for s in ids); Hs = sum(pr[s][2] for s in ids)
        base = dict(n=len(ids), excluded_no_blocks=[s for s in D.ids if s not in ids], per_person=dict(zip(ids, map(float, x))),
                    n_seeds_per_person={s: pr[s][3] for s in ids}, test_blocks_per_person={s: pr[s][4] for s in ids},
                    test_h_per_person={s: pr[s][2] for s in ids}, pooled_rate=float(Ns / Hs) if Hs else None)
        if len(ids) < 2:                                                      # technical guard: t test undefined
            out[rule] = dict(base, mean=float(x.mean()) if len(x) else None, t_p=None, reject=False, ci90=None, wilcoxon_p=None,
                             boot_percentile_q05=None, computable=False); continue
        tt = ttest_1samp(x, BOUND_R, alternative="greater"); half = tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        dd = x - BOUND_R
        out[rule] = dict(base, mean=float(x.mean()), t_p=float(tt.pvalue), reject=bool(tt.pvalue < 0.05),
                         ci90=[float(x.mean() - half), float(x.mean() + half)],
                         wilcoxon_p=float(wilcoxon(dd, alternative="greater").pvalue) if np.any(dd != 0) else None,
                         boot_percentile_q05=boot_q(x, 0.05, stat=np.mean), computable=True)
    r13 = cells_rule_on(D, "R", 13); t13 = cells_rule_on(D, "Rthin", 13); pp = {}
    for s in D.ids:                                                           # rev 8: seeds where both rules have blocks
        cr, ct = person_counts(r13, s), person_counts(t13, s); both = tuple(k for k in SEEDS if cr[k][1] > 0 and ct[k][1] > 0)
        if both:
            a, b = person_rate(r13, s, both), person_rate(t13, s, both)
            pp[s] = dict(R=a[0], Rthin=b[0], n_seeds=len(both), test_blocks=a[4])
    ids = list(pp); diff = np.array([pp[s]["R"] - pp[s]["Rthin"] for s in ids])
    if len(ids) >= 2:
        half = tdist.ppf(0.95, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        out["paired"] = dict(per_person=pp, n=len(ids), excluded_no_blocks=[s for s in D.ids if s not in pp],
                             diff_mean=float(diff.mean()), ci90=[float(diff.mean() - half), float(diff.mean() + half)], computable=True)
    else:
        out["paired"] = dict(per_person=pp, n=len(ids), excluded_no_blocks=[s for s in D.ids if s not in pp],
                             diff_mean=float(diff.mean()) if len(ids) else None, ci90=None, computable=False)
    out["acf"] = acf_summary(D)
    return out


def acf_summary(D):
    """b_rchecks.py B2: Fisher z pooled with weights n - 3, seed-averaged z; raw and linearly detrended."""
    def acf(x, lag):
        x = x - x.mean(); return float(np.sum(x[lag:] * x[:-lag]) / np.sum(x * x))
    res = {}
    for detr in (False, True):
        zw = {1: [0.0, 0.0], 2: [0.0, 0.0]}; per = {}
        for sid in D.ids:
            per[sid] = {}
            for lag in (1, 2):
                zs, ns = [], []
                for kk in range(3):
                    use = [k for k in SEEDS if len(D.S[(sid, k)][kk]) >= 8]   # rev 8: seeds with data; weight = max length
                    if not use: continue
                    n = max(len(D.S[(sid, k)][kk]) for k in use)
                    zz = []
                    for k in use:
                        S = D.S[(sid, k)][kk]; jx = np.arange(len(S))
                        x = S - np.polyval(np.polyfit(jx, S, 1), jx) if detr else S
                        zz.append(np.arctanh(np.clip(acf(x, lag), -0.999, 0.999)))
                    zs.append(np.mean(zz)); ns.append(n - 3)
                if ns:
                    zp = float(np.sum(np.array(zs) * ns) / sum(ns)); se = 1 / np.sqrt(sum(ns))
                    per[sid][f"lag{lag}"] = dict(r=float(np.tanh(zp)), ci90=[float(np.tanh(zp - 1.645 * se)), float(np.tanh(zp + 1.645 * se))])
                    zw[lag][0] += zp * sum(ns); zw[lag][1] += sum(ns)
        pooled = {f"lag{l}": dict(r=float(np.tanh(v[0] / v[1])), ci90=[float(np.tanh(v[0] / v[1] - 1.645 / np.sqrt(v[1]))), float(np.tanh(v[0] / v[1] + 1.645 / np.sqrt(v[1])))])
                  for l, v in zw.items() if v[1]}
        res["detrended" if detr else "raw"] = dict(pooled=pooled, per_person=per)
    return res


def sensitivity(D):
    """§4.4: injected events; K, R, R-thin, R-sel, ACI with clean-stream and self thresholds (controls.py,
    r3_checks.py, b4_aci.py); comparators main/B0/B0m/B0sm/B6 at matched FAR (analysis.py); per person."""
    # matched FAR over all IDs per seed (analysis.py)
    match = {}
    for k in SEEDS:
        tbm_all, tgt, tgt_k, calv = [], 0, {0: 0, 1: 0, 2: 0}, []
        for sid in D.ids:
            st = D.z[(sid, k)]["st"]; cl = D.cl[(sid, k)]; tau = M.main_thresholds(cl["S"], cl["nc"], N_CAL, KSTAR)
            tm = M.test_mask(st, cl["blk"], tau, cl["nc"], N_CAL); tbm = {}
            for i in np.flatnonzero(tm):
                key = (int(st[i]), int(cl["blk"][i])); tbm[key] = max(tbm.get(key, -np.inf), cl["D"][i])
            tbm_all.append(tbm)
            for (kk, b), mx in tbm.items():
                if tau[kk] is not None and mx > tau[kk]: tgt += 1; tgt_k[kk] += 1
            calv.append({kk: cl["D"][(st == kk) & (cl["blk"] >= 0) & (cl["blk"] < N_CAL) & np.isfinite(cl["D"])] for kk in range(3) if tau[kk] is not None})
        allmx = np.sort(np.concatenate([[mx for mx in t.values()] for t in tbm_all]))[::-1]
        c = float((allmx[tgt - 1] + allmx[tgt]) / 2) if 0 < tgt < len(allmx) else (float(allmx[0]) + 1 if tgt == 0 else -np.inf)
        ck = {}
        for kk in range(3):
            v = np.sort([mx for t in tbm_all for (s_, b), mx in t.items() if s_ == kk])[::-1]; tt = tgt_k[kk]
            ck[kk] = (float((v[tt - 1] + v[tt]) / 2) if 0 < tt < len(v) else (float(v[0]) + 1 if (tt == 0 and len(v)) else None)) if len(v) else None
        best = None
        for q in np.arange(0.50, 1.0001, 0.0025):
            cnt = 0
            for tbm, cv in zip(tbm_all, calv):
                t = {kk: float(np.quantile(v, q)) for kk, v in cv.items()}
                cnt += sum(1 for (kk, b), mx in tbm.items() if mx > t[kk])
            if best is None or abs(cnt - tgt) < abs(best[1] - tgt): best = (float(q), cnt)
        match[k] = dict(main_count=tgt, B0m_c=c, B0sm_c=ck, B6_q=best[0], B6_count=best[1])
    ev = []
    for (sid, k) in D.z:
        z = D.z[(sid, k)]; st = z["st"]; cl = D.cl[(sid, k)]
        tau = M.main_thresholds(cl["S"], cl["nc"], N_CAL, KSTAR)
        ce = {kk: cal_end_time(st, cl["blk"], kk) for kk in range(3)}
        per_clean = {rule: {} for rule in ("R", "Rthin", "Rsel", "ACI")}
        for kk in range(3):
            S = cl["S"][kk][:cl["nc"][kk]]
            for rule in per_clean: per_clean[rule].update({(kk, j): t for j, t in block_rule(S, len(S), rule).items()})
        b6 = {}
        for kk in range(3):
            if tau[kk] is None: b6[kk] = None; continue
            v = cl["D"][(st == kk) & (cl["blk"] >= 0) & (cl["blk"] < N_CAL) & np.isfinite(cl["D"])]; b6[kk] = float(np.quantile(v, match[k]["B6_q"]))
        for d in DELTAS:
            inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"])
            ons = [t for t in z[f"on{d}"] if tau[onset_state(st, t)] is not None and t > ce[onset_state(st, t)]]
            tm = M.test_mask(st, inj["blk"], tau, inj["nc"], N_CAL)
            det = {"K": detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau), tm), ons),
                   "B0": detect(M.alarms(inj["D"], st, inj["blk"], np.full(len(st), 120.0), tm), ons),
                   "B0m": detect(M.alarms(inj["D"], st, inj["blk"], np.full(len(st), match[k]["B0m_c"]), tm), ons),
                   "B0sm": detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, match[k]["B0sm_c"]), tm), ons),
                   "B6": detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, b6), tm), ons)}
            for rule, per in per_clean.items():
                th = np.array([per.get((int(st[i]), int(inj["blk"][i])), np.inf) for i in range(len(st))])
                det[rule] = detect(M.alarms(inj["D"], st, inj["blk"], th, np.isfinite(th) | (th == -np.inf)), ons)
            tau_s = M.main_thresholds(inj["S"], inj["nc"], N_CAL, KSTAR); tms = M.test_mask(st, inj["blk"], tau_s, inj["nc"], N_CAL)
            det["K_self"] = detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau_s), tms), ons)
            for rule in ("R", "Rthin", "Rsel", "ACI"):
                ps = {}
                for kk in range(3):
                    S = inj["S"][kk][:inj["nc"][kk]]; ps.update({(kk, j): t for j, t in block_rule(S, len(S), rule).items()})
                th = np.array([ps.get((int(st[i]), int(inj["blk"][i])), np.inf) for i in range(len(st))])
                det[rule + "_self"] = detect(M.alarms(inj["D"], st, inj["blk"], th, np.isfinite(th) | (th == -np.inf)), ons)
            for j, t0 in enumerate(ons):
                ev.append(dict(sid=sid, seed=k, delta=d, state=STN[onset_state(st, t0)], **{f"det_{m}": bool(v[j][0]) for m, v in det.items()},
                               **{f"delay_{m}": v[j][1] for m, v in det.items()}))
    methods = sorted({key[4:] for key in ev[0] if key.startswith("det_")}) if ev else []
    summ = {}
    for d in DELTAS:
        for stt in ("all", "rest", "light", "vigorous"):
            rows = [e for e in ev if e["delta"] == d and (stt == "all" or e["state"] == stt)]
            if not rows: continue
            summ[f"{d}|{stt}"] = dict(n_events=len(rows), **{f"sens_{m}": float(np.mean([e[f"det_{m}"] for e in rows])) for m in methods},
                                      **{f"delay_med_{m}": (float(np.nanmedian([e[f"delay_{m}"] for e in rows if e[f"det_{m}"]])) if any(e[f"det_{m}"] for e in rows) else None) for m in methods})
    per_person = {}
    for d, sid in itertools.product(DELTAS, D.ids):
        per_seed = [[e for e in ev if e["delta"] == d and e["sid"] == sid and e["seed"] == k] for k in SEEDS]
        used = [x for x in per_seed if x]                                     # rev 8: seeds with >= 1 event in the test region
        if used:
            per_person[f"{d}|{sid}"] = {m: float(np.mean([np.mean([e[f"det_{m}"] for e in x]) for x in used])) for m in methods}
            per_person[f"{d}|{sid}"]["n_seeds"] = len(used); per_person[f"{d}|{sid}"]["n_events_mean"] = float(np.mean([len(x) for x in used]))
    mdiff = {}
    for d in DELTAS:
        ids = [s for s in D.ids if f"{d}|{s}" in per_person]
        for comp in ("B0m", "B0sm"):
            v = np.array([per_person[f"{d}|{s}"]["K"] - per_person[f"{d}|{s}"][comp] for s in ids])
            if not len(v): mdiff[f"{d}|K-{comp}"] = dict(per_person={}, n=0, mean=None, ci95=None); continue
            g = np.random.default_rng(SEED_BOOT); bi = g.integers(0, len(v), (N_BOOT, len(v))); bm = v[bi].mean(1)
            mdiff[f"{d}|K-{comp}"] = dict(per_person=dict(zip(ids, map(float, v))), n=len(ids), mean=float(v.mean()),
                                          ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))])
    return dict(match={str(k): v for k, v in match.items()}, summary=summ, per_person=per_person, matched_diff=mdiff)


def masking(D, delta=30, n_draw=10, hmax=14):
    """§4.3 (controls.py / r3_checks.py / b4_aci.py): block-level simulation; detection of a second event h blocks
    after a first one, with vs without the first event, per rule."""
    g = np.random.default_rng(SEED_MASK); pools = {}
    for (sid, k) in D.z:
        z = D.z[(sid, k)]; st = z["st"]; inj = stream(st, z[f"sq{delta}"], z[f"hp3_{delta}"]); c = 2.0 * np.arange(len(st)) + 4
        for t in z[f"on{delta}"]:
            kk = onset_state(st, t); e = inj["D"][(c >= t) & (c <= t + WIN_DET) & (st == kk)]; e = e[np.isfinite(e)]
            if len(e): pools.setdefault((sid, kk, k), []).append(float(e.max()))
    res = {}
    for rule in ("K", "R", "Rthin", "Rsel", "ACI"):
        mk = {h: [0, 0, 0] for h in range(1, hmax + 1)}
        for (sid, k) in D.z:
            for kk, S in enumerate(D.S[(sid, k)]):
                nc = len(S); pool = np.array(pools.get((sid, kk, k), []))
                j_lo = FIRST_TEST[rule] + 1
                if nc < j_lo + hmax + 1 or len(pool) < 3: continue
                base = block_rule(S, nc, rule)
                for j0 in range(j_lo, nc - hmax):
                    for _ in range(n_draw):
                        e1, e2 = g.choice(pool), g.choice(pool); S1 = S.copy(); S1[j0] = max(S[j0], e1); alt = block_rule(S1, nc, rule)
                        for h in range(1, hmax + 1):
                            r = mk[h]; r[0] += e2 > base[j0 + h]; r[1] += e2 > alt[j0 + h]; r[2] += 1
        res[rule] = {h: dict(no_prior=v[0] / v[2], after=v[1] / v[2], n=v[2]) for h, v in mk.items() if v[2]}
    return res


def aci_bound(D):
    """r5_aci_bound.py: Gibbs & Candes 2021 Prop 4.1 per cell, alpha* = 1/7, gamma = 0.05."""
    rows = []
    for sid in D.ids:
        for kk in range(3):
            Ts = [len(D.S[(sid, k)][kk]) - N_CAL for k in SEEDS if len(D.S[(sid, k)][kk]) > N_CAL]   # rev 8
            if not Ts: continue
            T = float(np.mean(Ts))
            b = (max(ACI_ALPHA, 1 - ACI_ALPHA) + ACI_GAMMA) / (ACI_GAMMA * T) if T else np.inf
            lo, hi = max(ACI_ALPHA - b, 0.0), min(ACI_ALPHA + b, 1.0)
            rows.append(dict(sid=sid, state=STN[kk], T=T, bound=b, far_range=[6 * lo, 6 * hi], vacuous=bool(lo == 0 and hi == 1)))
    return dict(cells=rows, n_vacuous=sum(r["vacuous"] for r in rows))


def contamination(D):
    """analysis.py: one +30 event in calibration block 2 of each state; tau change and K sensitivity."""
    dt, sens = [], {d: [[], []] for d in DELTAS}
    for (sid, k) in D.z:
        z = D.z[(sid, k)]; st = z["st"]; cl = D.cl[(sid, k)]
        if len(z["on_cal"]) == 0: continue
        tau = M.main_thresholds(cl["S"], cl["nc"], N_CAL, KSTAR); ce = {kk: cal_end_time(st, cl["blk"], kk) for kk in range(3)}
        cc = stream(st, z["sq_cal"], z["hp3_cal"]); tau_c = M.main_thresholds(cc["S"], cc["nc"], N_CAL, KSTAR)
        for t0 in z["on_cal"]:
            kk = int(st[int(t0 // 2)])
            if tau[kk] is not None and tau_c[kk] is not None: dt.append(dict(sid=sid, seed=k, state=STN[kk], d_tau=tau_c[kk] - tau[kk]))
        tau_mix = {kk: (tau_c[kk] if tau_c[kk] is not None else tau[kk]) for kk in range(3)}
        for d in DELTAS:
            inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"]); tm = M.test_mask(st, inj["blk"], tau, inj["nc"], N_CAL)
            ons = [t for t in z[f"on{d}"] if tau[onset_state(st, t)] is not None and t > ce[onset_state(st, t)]]
            sens[d][0] += [x[0] for x in detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau), tm), ons)]
            sens[d][1] += [x[0] for x in detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau_mix), tm), ons)]
    return dict(d_tau=dt, sens={d: dict(clean_cal=float(np.mean(v[0])) if v[0] else None, contaminated_cal=float(np.mean(v[1])) if v[1] else None, n=len(v[0])) for d, v in sens.items()})


def coverage(D):
    """analysis.py m5 extended to R / R-thin: fraction of wear time with an active within-person threshold."""
    out = []
    for sid in D.ids:
        z = D.z[(sid, 0)]; st = z["st"]; cl = D.cl[(sid, 0)]; n = len(st)
        for kk in range(3):
            mon_k = cl["mon"] & (st == kk); row = dict(sid=sid, state=STN[kk], monitored_h=float(mon_k.sum() * 2 / 3600), complete_blocks=cl["nc"][kk])
            for rule in ("K", "R", "Rthin"):
                first = FIRST_TEST[rule]; act = mon_k & (cl["blk"] >= first) & (cl["blk"] < cl["nc"][kk]) if cl["nc"][kk] > first else np.zeros(n, bool)
                idx = np.flatnonzero((st == kk) & (cl["blk"] == first - 1))
                row[f"{rule}_frac_of_wear"] = float(act.sum() / n); row[f"{rule}_cal_done_h"] = float((2.0 * idx[-1] + 4) / 3600) if (len(idx) and cl["nc"][kk] > first) else None
            out.append(row)
    per_person = {sid: {rule: float(sum(r[f"{rule}_frac_of_wear"] for r in out if r["sid"] == sid)) for rule in ("K", "R", "Rthin")} for sid in D.ids}
    return dict(cells=out, per_person_frac_wear=per_person)


# =========================================================================== part 4: conclusion sentences
L1 = ("All participants of this dataset followed the same one-day route with an altitude change of about 3,000 m, so the "
      "persons are not independent in the direction of any change over the day, and the result may not carry over to "
      "free-living wear without such a change.")


def conclusions(res):
    """Fill the fixed sentences (prereg rev 7 §4.1, 4.1b, 4.2, 4.3). Returns a list of dicts
    (table, branch, reason, sentence); every sentence is followed by the L1 scope sentence."""
    out = []; hk = res["H_K"]; f = lambda x: f"{x:.2f}"

    def add(table, branch, reason, sentence):
        out.append(dict(table=table, branch=branch, reason=reason, sentence=sentence + " " + L1))
    n_c = hk["n_cells"]
    if hk["X2_reject"]:
        add("H-K", "rejected", f"X2 bootstrap 5 % quantile {f(hk['X2_q05'])} > 0",
            f"Under free-living wear (WildPPG, {len(res['ids'])} held-out persons), {f(hk['Xbar'])} of {n_c} (person, state) cells exceeded 2 "
            f"false alarms per monitored hour (seed average), more than the {f(hk['E0bar'])} expected if the per-cell tolerance guarantee held "
            f"(person-cluster bootstrap lower bound > 0). The guarantee of Proposition 2a did not transfer to these data. Its assumption "
            f"A(0) is violated in at least part of the cells. §4.2 states which source of the violation the data support.")
    else:
        mean_part = (f" The mean rate was {f(hk['mean_rate_of_persons'])} per monitored hour against 0.857 expected under A(0) "
                     f"(Y2: bootstrap 5 % quantile {f(hk['Y2_q05'])}).")
        tail = (" This was against the dev-based prediction. It does not prove the guarantee: power was about 0.75 when 20 % of "
                "cells run at 1.5 r (simulation, §4.6, not adjusted for a drift shared by all persons).")
        if hk["Y2_reject"]:
            add("H-K", "not rejected; Y2 rejected", f"X2 quantile {f(hk['X2_q05'])} ≤ 0, Y2 quantile {f(hk['Y2_q05'])} > 0",
                f"Under free-living wear, {f(hk['Xbar'])} of {n_c} cells exceeded 2 false alarms per monitored hour, against the "
                f"{f(hk['E0bar'])} expected under the per-cell tolerance guarantee (bootstrap lower bound ≤ 0)." + mean_part +
                " The tolerance statement was not refuted, but the mean rate exceeded its expectation under A(0)." + tail)
        else:
            add("H-K", "not rejected; Y2 not rejected", f"X2 quantile {f(hk['X2_q05'])} ≤ 0, Y2 quantile {f(hk['Y2_q05'])} ≤ 0",
                f"Under free-living wear, {f(hk['Xbar'])} of {n_c} cells exceeded 2 false alarms per monitored hour, consistent with "
                f"the {f(hk['E0bar'])} expected under the per-cell tolerance guarantee (bootstrap lower bound ≤ 0)." + mean_part + tail)
    k1 = res["H_K1"]
    if hk["X2_reject"] and k1["n"] == 0:                                      # rev 8 technical guard
        add("H-K1", "not computable (n = 0)", "H-K rejected; no person has a test block of K with m = 1 in any seed",
            "H-K1 could not be computed: no person had a test block of K with m = 1 (n = 0).")
    elif not hk["X2_reject"]:
        out.append(dict(table="H-K1", branch="not tested (fixed sequence)", reason="H-K not rejected",
                        sentence=f"H-K1 was not tested (fixed sequence). Its statistic is description only: sum d = {f(k1['sum_d'])}, "
                                 + (f"bootstrap 5 % quantile {f(k1['q05'])}." if k1["q05"] is not None else "no bootstrap (n = 0).")))
    elif k1["reject"]:
        add("H-K1", "rejected", f"H-K rejected; H-K1 quantile {f(k1['q05'])} > 0", "The excess remains when lag-1 dependence is allowed.")
    else:
        add("H-K1", "not rejected", f"H-K rejected; H-K1 quantile {f(k1['q05'])} ≤ 0",
            "The excess under K is compatible with short-range dependence between neighbouring blocks; with thinned calibration the "
            f"expectation level was not refuted at n = {k1['n']}.")
    po = res["p_order"]["p"]; cov = res["covariates"]; pd = cov["primary_dalt"]
    prim_ok = not cov["bootstrap_dropped"]["dalt"]["descriptive_only"]
    if not hk["X2_reject"]:
        out.append(dict(table="attribution", branch="not applied", reason="H-K not rejected", sentence="(no attribution sentence)"))
    elif po >= 0.05:
        add("attribution", "row 1", f"p_order {po:.3f} ≥ 0.05", "The data do not show that the time order of the blocks is the source of the violation.")
    elif not prim_ok or pd["ci90"][0] <= 0:
        why = "primary model descriptive only (> 5 % dropped draws)" if not prim_ok else f"dAlt 90 % CI {pd['ci90']} not above 0"
        add("attribution", "row 2", f"p_order {po:.3f} < 0.05; {why}",
            "Block scores are not exchangeable in time order (serial dependence and/or drift); exceedances were not associated with altitude change.")
    else:
        s = ("Block scores are not exchangeable in time order, and exceedances were associated with altitude change since calibration "
             "(non-stationarity over the day); physiological and behavioural components were not separated.")
        reason = f"p_order {po:.3f} < 0.05; dAlt 90 % CI {[round(x, 2) for x in pd['ci90']]} above 0"
        s2 = cov["sens2_dalt_dtime"]["dalt"]["ci90"]
        if s2[0] <= 0 <= s2[1]:
            s += " Altitude change and time elapsed since calibration could not be separated."; reason += "; dAlt CI with dTime contains 0"
        add("attribution", "row 3", reason, s)
    hr = res["H_R"]["R"]
    if not hr.get("computable", True):                                        # rev 8 technical guard
        add("H-R", f"not computable (n = {hr['n']})", f"fewer than 2 persons with R test blocks (n = {hr['n']})",
            f"The H-R test could not be computed: fewer than 2 persons had test blocks of R (n = {hr['n']}).")
    elif hr["reject"]:
        add("H-R", "rejected", f"t p {hr['t_p']:.3f} < 0.05",
            "Rolling calibration did not keep the expectation bound: the 7 blocks of a window were not exchangeable. This does not by "
            "itself indicate drift. Stationary short-range dependence between neighbouring blocks (lag 1) is enough to raise the "
            "exceedance probability above 1/7 (R1.md Remark 3: MA(1) gives 1.04–1.10/h). The R-thin result is reported next to it.")
    else:
        add("H-R", "not rejected", f"t p {hr['t_p']:.3f} ≥ 0.05",
            f"Rolling calibration kept the long-run rate consistent with its expectation bound of 0.857/h (mean of persons {f(hr['mean'])}, "
            f"90 % CI {f(hr['ci90'][0])}–{f(hr['ci90'][1])}): not refuted at n = {hr['n']}, which is not the same as \"the bound holds\". "
            "Rolling calibration gives only an expectation bound, not the per-cell tolerance guarantee.")
    pr = res["H_R"]["paired"]
    if not pr.get("computable", True):                                        # rev 8 technical guard
        add("R vs R-thin", f"not computable (n = {pr['n']})", f"fewer than 2 persons with test blocks of both rules (n = {pr['n']})",
            f"The paired comparison could not be computed: fewer than 2 persons had test blocks of both R and R-thin (n = {pr['n']}).")
    elif pr["ci90"][0] > 0:
        add("R vs R-thin", "R − R-thin > 0", f"90 % CI {[round(x, 2) for x in pr['ci90']]} above 0",
            "On the same test blocks, the contiguous rolling window gave more false alarms than the spread window, as short-range "
            "dependence between neighbouring blocks predicts; this does not exclude drift.")
    elif pr["ci90"][1] < 0:
        add("R vs R-thin", "R − R-thin < 0", f"90 % CI {[round(x, 2) for x in pr['ci90']]} below 0",
            "On the same test blocks, the spread window gave at least as many false alarms, as an upward change over the day would "
            "predict and short-range dependence alone would not.")
    else:
        add("R vs R-thin", "CI contains 0", f"90 % CI {[round(x, 2) for x in pr['ci90']]} contains 0",
            "The paired comparison does not separate short-range dependence from non-stationarity.")
    return out


# =========================================================================== part 5: run everything
def analyse(ids, cache_dir=None, log=print, excluded=None, D=None):
    """excluded: {sid: reason} of persons whose raw file was unreadable (rev 8); they are not in ids and none of their
    arrays is loaded. D: prebuilt Data (tests only)."""
    excluded = dict(excluded or {})
    D = D if D is not None else Data(ids, cache_dir)
    res = dict(ids=list(ids), excluded_unreadable=excluded,
               deviations=[f"{s}: raw file unreadable after 2 attempts ({r}); excluded from every analysis (prereg rev 8 §4.0)" for s, r in excluded.items()])
    steps = [("H_K", h_k), ("H_K1", h_k1), ("p_order", p_order), ("descriptive_drift", descriptive_drift), ("covariates", covariates),
             ("H_R", h_r), ("FAR", lambda D: {rule: far_summary(D, rule) for rule in ("K", "Km1", "R", "Rthin", "Rsel", "ACI")}),
             ("sensitivity", sensitivity), ("masking", masking), ("ACI_bound", aci_bound), ("contamination", contamination), ("coverage", coverage)]
    for name, fn in steps:
        res[name] = fn(D); log(f"  analysed: {name}")
    res["conclusions"] = conclusions(res)
    return res
