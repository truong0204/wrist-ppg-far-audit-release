"""S1c — signal-quality gate (SQI) for E2, chosen on WildPPG DEVELOPMENT IDs only (e61, u7y, ngh).
PI decision 22-11 (after the zero-shot E2 failed the injection test at +40 on WildPPG dev): add an SQI gate.
Becomes deviation #6 in prereg v2.2. Confirmatory PPG stays locked.

    uv run python 05_experiments/s1c/sqi_dev.py

Definitions (fixed before running):
- E2 is unchanged: zero-shot all-DaLiA SpecFCN + causal HMM, sigma = 3 bpm, run on every window.
- SQI_i = max softmax probability of the SpecFCN output at window i (reference-free, PPG + ACC only, causal).
- Window i is MONITORED iff SQI_i >= theta. Decisions (alarms) use monitored windows only; FAR is per
  MONITORED hour = (#monitored windows) * 2 s / 3600. Coverage = monitored / all windows.
- Gated injection test: recovered = median E2 over monitored plateau windows - median E2 over monitored windows
  in the 60 s before onset. An event is evaluable iff >= 50 % of its plateau and >= 50 % of its pre-window
  windows are monitored. Pass iff, for each Delta in {20, 30, 40}, the median recovered over evaluable events
  >= Delta / 2 (amendment 1 criterion, applied to monitored data).
- theta candidates = quantiles q in {0.10, 0.15, ..., 0.50} of SQI pooled over the 3 dev IDs x 5 seeds on
  unmodified data. Rule: choose the SMALLEST q that passes (max coverage); freeze its absolute theta (seed-pooled).
  If no q <= 0.50 passes: stop and report (no gate retaining >= half the data works).
Also reported (information only, no choice made from it): coverage, gated B0 FAR_any per monitored hour, and the
gated injection test on DaLiA dev S10-S12 with the LOSO dev models.
"""
import sys, json, time, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W, io_dalia, e2cnn, features, events, stats, inject
torch.set_num_threads(8)
R = ROOT / "05_experiments/s1c/results"; SEEDS = [0, 1, 2, 3, 4]; SIGMA = 3.0; GRID = e2cnn.FCN_GRID
QS = [round(0.10 + 0.05 * i, 2) for i in range(9)]
t_start = time.time()


def model(path):
    m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(path)); return m.eval()


def run_injection(ppg64, acc32, ref2, lev, m, k):
    """Injected runs for Delta = 20/30/40: returns {delta: (onsets, hp, sqi)}."""
    n = len(lev); out = {}
    for delta in (20, 30, 40):
        rng = np.random.default_rng(1000 * k + delta)
        onsets = inject.schedule(n, lev, rng, per_hour=3, min_gap_s=600, dur_s=240, start_s=120)
        ppg = ppg64.copy(); r = ref2.copy()
        for t0 in onsets:
            ppg, r, _ = inject.inject(ppg, 64, r, t0, delta, ramp_s=30, plateau_s=180)
        P = e2cnn.predict_proba(m, features.spectra(ppg, acc32, n))
        out[delta] = (onsets, e2cnn.hmm_track_grid(P, GRID, SIGMA).astype(np.float32), P.max(1).astype(np.float32))
    return out


def gated_injection(runs, theta):
    res = {}
    for delta in (20, 30, 40):
        rec, n_all = [], 0
        for onsets, hp, sq in runs[delta]:
            c = np.arange(len(hp)) * 2 + 4.0; mon = sq >= theta
            for t0 in onsets:
                n_all += 1
                pl = (c > t0 + 30) & (c < t0 + 210); pre = (c > t0 - 60) & (c < t0)
                if mon[pl].mean() >= 0.5 and mon[pre].mean() >= 0.5:
                    rec.append(float(np.median(hp[pl & mon]) - np.median(hp[pre & mon])))
        v = np.array(rec)
        res[delta] = dict(n_events=n_all, n_evaluable=len(v), frac_evaluable=len(v) / max(n_all, 1),
                          recovered_median=float(np.median(v)) if len(v) else None,
                          frac_ge_half=float(np.mean(v >= delta / 2)) if len(v) else None,
                          passed=bool(len(v) and np.median(v) >= delta / 2))
    return res


def prep(ppg64, acc32, n):
    lev = np.array([np.std(np.linalg.norm(acc32[i * 64:i * 64 + 256], axis=1)) for i in range(n)]) < 0.10
    return lev


# ---------- WildPPG dev ----------
WD = {}
for s in W.DEV:
    d = W.load_wrist(s)
    ppg64, acc32 = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
    n = (len(ppg64) // 64 - 8) // 2 + 1
    X = features.spectra(ppg64, acc32, n); lev = prep(ppg64, acc32, n)
    ref2 = np.repeat(d["hr_ref"], 4)[:n]; ref2 = np.concatenate([ref2, np.full(n - len(ref2), np.nan)])
    WD[s] = dict(n=n, lev=lev, clean={}, inj={})
    for k in SEEDS:
        m = model(R / f"fcn_all_seed{k}.pt")
        P = e2cnn.predict_proba(m, X)
        WD[s]["clean"][k] = (e2cnn.hmm_track_grid(P, GRID, SIGMA), P.max(1))
        WD[s]["inj"][k] = run_injection(ppg64, acc32, ref2, lev, m, k)
    print("WildPPG", s, "done %.0f s" % (time.time() - t_start), flush=True)

pooled = np.concatenate([WD[s]["clean"][k][1] for s in W.DEV for k in SEEDS])
cand = {q: float(np.quantile(pooled, q)) for q in QS}
runs_w = {dl: [WD[s]["inj"][k][dl] for s in W.DEV for k in SEEDS] for dl in (20, 30, 40)}
scan = {}
for q, th in cand.items():
    g = gated_injection(runs_w, th)
    cov = {s: float(np.mean([np.mean(WD[s]["clean"][k][1] >= th) for k in SEEDS])) for s in W.DEV}
    scan[q] = dict(theta=th, coverage=cov, injection=g, passed=all(g[dl]["passed"] for dl in g))
    print("q=%.2f theta=%.4f cov %s | " % (q, th, {a: round(b, 2) for a, b in cov.items()}) + " ".join(
        "+%d: %s (eval %.2f)" % (dl, None if g[dl]["recovered_median"] is None else round(g[dl]["recovered_median"], 1),
                                 g[dl]["frac_evaluable"]) for dl in g) + (" PASS" if scan[q]["passed"] else ""), flush=True)

chosen = next((q for q in QS if scan[q]["passed"]), None)
out = dict(qs=QS, scan=scan, chosen_q=chosen, sigma=SIGMA, model="fcn_all_seed{0..4}.pt (zero-shot)")
if chosen is None:
    print("STOP: no q <= 0.50 passes the gated injection test")
else:
    th = scan[chosen]["theta"]; out["theta"] = th
    # gated B0 (> 120 bpm), in-control framing: every alarm on unmodified data counts
    kb = []
    for k in SEEDS:
        n_al = 0; mon_h = 0.0
        for s in W.DEV:
            hp, sq = WD[s]["clean"][k]; mon = sq >= th
            n_al += len(events.episodes((hp > 120) & mon, 3, 5)); mon_h += mon.sum() * 2 / 3600
        lo, hi = stats.poisson_rate_ci(n_al, mon_h)[1:]
        kb.append(dict(seed=k, n_alarms=n_al, monitored_hours=mon_h, n_subjects=3, far_any=n_al / mon_h, ci=[lo, hi]))
        print("gated B0 seed %d: FAR_any %.2f/h [%.2f, %.2f] (%d alarms, %.1f monitored h, 3 IDs)" % (k, n_al / mon_h, lo, hi, n_al, mon_h))
    out["kb_gated"] = kb
    # DaLiA dev, information only
    runs_d = {dl: [] for dl in (20, 30, 40)}; cov_d = {}
    Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz")
    for s in ("S10", "S11", "S12"):
        dd = io_dalia.load_subject(s); n = len(dd["label"]); lev = prep(dd["bvp"], dd["acc"], n)
        cs = []
        for k in SEEDS:
            m = model(R / f"fcn_model_{s}_seed{k}.pt")
            cs.append(float(np.mean(e2cnn.predict_proba(m, Z[f"{s}_X"]).max(1) >= th)))
            r = run_injection(dd["bvp"], dd["acc"], dd["label"].astype(float), lev, m, k)
            for dl in r: runs_d[dl].append(r[dl])
        cov_d[s] = float(np.mean(cs))
    out["dalia_dev"] = dict(coverage=cov_d, injection=gated_injection(runs_d, th))
    print("DaLiA dev coverage", {a: round(b, 2) for a, b in cov_d.items()}, "gated injection",
          {dl: (round(v["recovered_median"], 1), round(v["frac_evaluable"], 2)) for dl, v in out["dalia_dev"]["injection"].items()})
out["runtime_s"] = time.time() - t_start
json.dump(out, open(R / "sqi_dev.json", "w"), indent=2)
print("chosen q", chosen, "theta", out.get("theta"), "runtime %.0f s" % out["runtime_s"])
