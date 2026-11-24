"""S1c domain check of E2 on WildPPG DEVELOPMENT IDs only (e61, u7y, ngh). Confirmatory PPG stays locked
(io_wildppg.load_wrist raises PermissionError for the 13 confirmatory IDs).

    uv run python 05_experiments/s1c/wildppg_dev_check.py zeroshot   # all-DaLiA SpecFCN, seeds 0-4
    uv run python 05_experiments/s1c/wildppg_dev_check.py finetune   # + fine-tune, leave-one-dev-ID-out

Pipeline: wrist ppg_g 128 Hz -> 64 Hz, ACC 128 Hz -> 32 Hz (features.to_dalia_rates); spectra on 8 s windows,
2 s stride; E2 = SpecFCN + causal HMM, sigma = 3 bpm (DaLiA-dev LOO, results/sigma_grid_ext.json).
Reference: authors' HR per non-overlapping 8 s window j (n_ref = duration / 8), compared with E2 window
i = 4j (same span [8j, 8j + 8] s). A lag scan (+-8 windows of 2 s) checks this alignment.

Decision rule, fixed before running (v2.1 §3: "optionally fine-tuned on WildPPG dev ... from WildPPG-dev MAE
only"): use the fine-tuned E2 only if (a) its leave-one-dev-ID-out MAE, pooled over the 3 IDs and 5 seeds, is
lower than zero-shot by >= 1.0 bpm, and (b) it passes the injection test (median recovered >= Delta/2 for
Delta = 20/30/40). Otherwise keep zero-shot (fewer choices). Fine-tune settings: init = all-DaLiA weights,
3 epochs, lr 1e-4, training windows = the 4j windows with a valid reference of the 2 other dev IDs.
"""
import sys, json, time, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W, e2cnn, features, events, stats, inject
torch.set_num_threads(8)
MODE = sys.argv[1] if len(sys.argv) > 1 else "zeroshot"
assert MODE in ("zeroshot", "finetune")
R = ROOT / "05_experiments/s1c/results"; SEEDS = [0, 1, 2, 3, 4]; SIGMA = 3.0; GRID = e2cnn.FCN_GRID
FT = dict(epochs=3, lr=1e-4)
t_start = time.time()

D = {}
for s in W.DEV:
    d = W.load_wrist(s)
    ppg64, acc32 = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
    n = (len(ppg64) // 64 - 8) // 2 + 1
    X = features.spectra(ppg64, acc32, n)
    ref = d["hr_ref"]; nj = min(len(ref), (len(X) - 1) // 4 + 1)
    lev = np.array([np.std(np.linalg.norm(acc32[i * 64:i * 64 + 256], axis=1)) for i in range(len(X))]) < 0.10
    D[s] = dict(X=X, ppg64=ppg64, acc32=acc32, ref=ref[:nj], nj=nj, lev=lev, hours=((len(X) - 1) * 2 + 8) / 3600)
    print(s, "windows", len(X), "ref", nj, "hours %.2f" % D[s]["hours"], "rest/light frac %.2f" % lev.mean(), flush=True)


def models_for(test):
    """Return {seed: state_dict} to evaluate on dev ID `test`."""
    out = {}
    for k in SEEDS:
        sd = torch.load(R / f"fcn_all_seed{k}.pt")
        if MODE == "finetune":
            tr = [o for o in W.DEV if o != test]
            Xs, ys = [], []
            for o in tr:
                j = np.flatnonzero(~np.isnan(D[o]["ref"])); Xs.append(D[o]["X"][4 * j]); ys.append(D[o]["ref"][j])
            m = e2cnn.train_fcn(np.concatenate(Xs), np.concatenate(ys), k, init=sd, **FT)
            sd = m.state_dict()
        out[k] = sd
    return out


def load(sd):
    m = e2cnn.SpecFCN(); m.load_state_dict(sd); return m.eval()


res = dict(mode=MODE, sigma=SIGMA, finetune=FT if MODE == "finetune" else None, per_id={}, kb=[], injection={})
TRK = {}
MODELS = {}
for s in W.DEV:
    MODELS[s] = models_for(s)
    ref, nj = D[s]["ref"], D[s]["nj"]; ok = ~np.isnan(ref); j = np.flatnonzero(ok)
    rows = []
    for k in SEEDS:
        P = e2cnn.predict_proba(load(MODELS[s][k]), D[s]["X"])
        hp = e2cnn.hmm_track_grid(P, GRID, SIGMA); TRK[(s, k)] = hp
        arg = GRID[P.argmax(1)]
        lev_j = D[s]["lev"][4 * j]
        lag = {L: float(np.mean(np.abs(hp[np.clip(4 * j + L, 0, len(hp) - 1)] - ref[j]))) for L in range(-8, 9)}
        rows.append(dict(seed=k, mae_argmax=float(np.mean(np.abs(arg[4 * j] - ref[j]))),
                         mae_e2=float(np.mean(np.abs(hp[4 * j] - ref[j]))),
                         mae_e2_rest=float(np.mean(np.abs(hp[4 * j][lev_j] - ref[j][lev_j]))),
                         mae_e2_motion=float(np.mean(np.abs(hp[4 * j][~lev_j] - ref[j][~lev_j]))),
                         best_lag_2s=int(min(lag, key=lag.get)), lag_mae=lag))
    res["per_id"][s] = dict(n_ref_valid=int(ok.sum()), hours=D[s]["hours"], frac_rest_light=float(D[s]["lev"].mean()), seeds=rows)
    print(s, "MAE argmax %.2f  E2 %.2f ± %.2f (rest %.2f, motion %.2f)  best lag %s" % (
        np.mean([r["mae_argmax"] for r in rows]), np.mean([r["mae_e2"] for r in rows]), np.std([r["mae_e2"] for r in rows]),
        np.mean([r["mae_e2_rest"] for r in rows]), np.mean([r["mae_e2_motion"] for r in rows]),
        [r["best_lag_2s"] for r in rows]), flush=True)

# B0 (> 120 bpm) on E2: FAR_any (primary, in-control framing) and FA vs reference episodes (ref upsampled to 2 s)
H = sum(D[s]["hours"] for s in W.DEV)
for k in SEEDS:
    fa = det = nref = n_al = 0
    for s in W.DEV:
        hp = TRK[(s, k)]; r2 = np.repeat(D[s]["ref"], 4)[:len(hp)]
        r2 = np.concatenate([r2, np.full(len(hp) - len(r2), np.nan)])
        ref_ep = events.episodes(np.nan_to_num(r2) > 120, 3, 5); al = events.episodes(hp > 120, 3, 5)
        f, d_ = events.match(al, ref_ep, 10); fa += f; det += d_; nref += len(ref_ep); n_al += len(al)
    lo, hi = stats.poisson_rate_ci(n_al, H)[1:]
    res["kb"].append(dict(seed=k, hours=H, n_subjects=3, n_alarms=n_al, far_any=n_al / H, far_any_ci=[lo, hi],
                          n_fa_matched=fa, far_matched=fa / H, det=det, n_ref=nref))
    print("B0 seed", k, "FAR_any %.2f/h [%.2f, %.2f] (%d alarms, %.1f h, 3 IDs); matched FA %.2f/h, det %d/%d" % (
        n_al / H, lo, hi, n_al, H, fa / H, det, nref), flush=True)

# Injection-recovery test (mandatory, amendment 1)
for delta in (20, 30, 40):
    rec = []
    for s in W.DEV:
        n = len(D[s]["X"]); ref2 = np.repeat(D[s]["ref"], 4)[:n]
        ref2 = np.concatenate([ref2, np.full(n - len(ref2), np.nan)])
        for k in SEEDS:
            rng = np.random.default_rng(1000 * k + delta)
            onsets = inject.schedule(n, D[s]["lev"], rng, per_hour=3, min_gap_s=600, dur_s=240, start_s=120)
            ppg = D[s]["ppg64"].copy(); r = ref2.copy()
            for t0 in onsets:
                ppg, r, _ = inject.inject(ppg, 64, r, t0, delta, ramp_s=30, plateau_s=180)
            hp = e2cnn.hmm_track_grid(e2cnn.predict_proba(load(MODELS[s][k]), features.spectra(ppg, D[s]["acc32"], n)), GRID, SIGMA)
            c = np.arange(len(hp)) * 2 + 4.0
            for t0 in onsets:
                pl = (c > t0 + 30) & (c < t0 + 210); pre = (c > t0 - 60) & (c < t0)
                rec.append(dict(subject=s, seed=k, t0=t0, recovered=float(np.median(hp[pl]) - np.median(hp[pre]))))
    v = np.array([x["recovered"] for x in rec])
    res["injection"][delta] = dict(n_events=len(rec), recovered_median=float(np.median(v)),
                                   recovered_iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                                   frac_ge_half=float(np.mean(v >= delta / 2)), passed=bool(np.median(v) >= delta / 2), events=rec)
    print("inject +%d: n=%d median %.1f IQR [%.1f, %.1f] frac>=half %.2f" % (
        delta, len(rec), np.median(v), *res["injection"][delta]["recovered_iqr"], np.mean(v >= delta / 2)), flush=True)

pooled = [r["mae_e2"] for s in W.DEV for r in res["per_id"][s]["seeds"]]
res["mae_e2_pooled_mean"] = float(np.mean(pooled)); res["runtime_s"] = time.time() - t_start
res["versions"] = dict(numpy=np.__version__, torch=torch.__version__)
json.dump(res, open(R / f"wildppg_dev_{MODE}.json", "w"), indent=2)
print("pooled E2 MAE %.2f; runtime %.0f s" % (res["mae_e2_pooled_mean"], res["runtime_s"]))
