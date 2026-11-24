"""S2 r1 M2 supplement (DEV ONLY, WildPPG e61/u7y/ngh, sigma 3, seeds 0-4). EXPLORATORY, post hoc.
(a) per (person, state): main-method observed FA rate on test blocks, and calibration-vs-test exchangeability
    check: permutation p of mean rank of the 6 calibration block maxima among all complete blocks
    (one-sided: calibration maxima LOWER than later blocks), 5,000 permutations.
(b) dev value of the M3 statistic X (cells with observed rate > 2/h) and its exact null tail (independent
    cells, per-cell P0 from Beta(1,6)-binomial with the cell's own number of test blocks).
(c) candidate v2.2 rule "rolling calibration": tau_j = max of the 6 most recent complete blocks of the same
    state ending g = 1 block before block j (buffer), recomputed each block; same debounce/refractory/test
    blocks as main (test blocks j >= 7). Threshold computed on the clean stream (as main); FAR on clean stream,
    sensitivity on injected streams (Delta 20/30/40, eligible events as in analysis.py).
    uv run python 05_experiments/s2/drift_extra.py
"""
import sys, json, pathlib, itertools, numpy as np
from scipy.stats import betabinom
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; N_CAL = 6; G = 1; WIN = 270.0; STN = ["rest", "light", "vigorous"]
rng = np.random.default_rng(20261122)


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return mon, blk, D, M.block_scores(D, st, blk), M.complete_blocks(mon, st, blk)


def rolling_thr(st, blk, S, nc):
    thr = np.full(len(st), np.inf); tm = np.zeros(len(st), bool)
    for k in range(3):
        for j in range(N_CAL + G, nc[k]):
            w = S[k][j - G - N_CAL:j - G]
            sel = (st == k) & (blk == j); thr[sel] = np.max(w); tm |= sel
    return thr, tm


def detect(a, ons):
    c = 2.0 * a + 4
    return [bool(np.any((c >= t) & (c <= t + WIN))) for t in ons]


cells, roll, sens = [], [], {m: {d: [] for d in (20, 30, 40)} for m in ("main", "rolling")}
for sid, s in itertools.product(("e61", "u7y", "ngh"), range(5)):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]
    mon, blk, D, S, nc = stream(st, z["sq"], z["hp3"])
    tau = M.main_thresholds(S, nc)
    rthr, rtm = rolling_thr(st, blk, S, nc)
    for k in range(3):
        if tau[k] is None: continue
        t1 = {kk: (tau[kk] if kk == k else None) for kk in range(3)}
        tm = M.test_mask(st, blk, t1, nc); a = M.alarms(D, st, blk, M.thr_from_state(st, t1), tm)
        T = nc[k] - N_CAL; h = M.monitored_hours(mon, tm)
        sk = S[k][:nc[k]]; rk = np.argsort(np.argsort(sk)); obs = rk[:N_CAL].mean()
        perm = np.array([rng.permutation(rk)[:N_CAL].mean() for _ in range(5000)])
        p_low = float((np.sum(perm <= obs) + 1) / 5001)
        rm = rtm & (st == k); ra = M.alarms(D, st, blk, rthr, rm); rh = M.monitored_hours(mon, rm)
        cells.append(dict(sid=sid, seed=s, state=STN[k], T=int(T), test_h=h, n_fa=int(len(a)), rate=len(a) / h,
                          cal_rank_p_low=p_low, roll_n_fa=int(len(ra)), roll_h=rh, roll_rate=len(ra) / rh if rh else None))
    # sensitivity (events in states with a main threshold, after main calibration; same events for both rules)
    ce = {k: (2.0 * np.flatnonzero((st == k) & (blk == N_CAL - 1))[-1] + 4 if tau[k] is not None else np.inf) for k in range(3)}
    for d in (20, 30, 40):
        m2, b2, D2, _, nc2 = stream(st, z[f"sq{d}"], z[f"hp3_{d}"])
        ons = [t for t in z[f"on{d}"] if tau[int(st[min(int((t - 4) // 2), len(st) - 1)])] is not None
               and t > ce[int(st[min(int((t - 4) // 2), len(st) - 1)])]]
        tmm = M.test_mask(st, b2, tau, nc2)
        sens["main"][d] += detect(M.alarms(D2, st, b2, M.thr_from_state(st, tau), tmm), ons)
        # rolling: thresholds from the clean stream, keyed by (state, block) of the clean stream; apply per window
        key = {(int(st[i]), int(blk[i])): rthr[i] for i in np.flatnonzero(rtm)}
        thr2 = np.array([key.get((int(st[i]), int(b2[i])), np.inf) for i in range(len(st))])
        sens["rolling"][d] += detect(M.alarms(D2, st, b2, thr2, np.isfinite(thr2)), ons)

# (b) dev X per seed and its exact null tail
xs = []
for s in range(5):
    cs = [c for c in cells if c["seed"] == s]
    X = sum(c["rate"] > 2 for c in cs)
    p0 = []
    for c in cs:
        n = np.arange(c["T"] + 1); pm = betabinom.pmf(n, c["T"], 1, 6); p0.append(pm[6 * n / c["T"] > 2].sum())
    dist = np.array([1.0])
    for p in p0: dist = np.convolve(dist, [1 - p, p])
    xs.append(dict(seed=s, X=int(X), n_cells=len(cs), EX0=float(sum(p0)), p_upper=float(dist[X:].sum())))
out = dict(cells=cells, dev_X=xs,
           sens={m: {d: dict(n=len(v), sens=float(np.mean(v))) for d, v in dd.items()} for m, dd in sens.items()},
           far_pooled=dict(main=sum(c["n_fa"] for c in cells) / sum(c["test_h"] for c in cells),
                           main_h=sum(c["test_h"] for c in cells),
                           rolling=sum(c["roll_n_fa"] for c in cells) / sum(c["roll_h"] for c in cells),
                           rolling_h=sum(c["roll_h"] for c in cells)))
json.dump(out, open(OUT / "drift_extra.json", "w"), indent=1)
for c in cells:
    if c["seed"] == 0 or True:
        pass
agg = {}
for c in cells:
    agg.setdefault((c["sid"], c["state"]), []).append(c)
for (sid, stn), v in agg.items():
    print("%s %-8s T=%2d main FA/h %s | cal-low p %s | rolling FA/h %s" % (
        sid, stn, v[0]["T"], [round(c["rate"], 2) for c in v], [round(c["cal_rank_p_low"], 3) for c in v],
        [None if c["roll_rate"] is None else round(c["roll_rate"], 2) for c in v]))
print("dev X:", xs)
print("pooled FAR", out["far_pooled"]); print("sens", out["sens"])
