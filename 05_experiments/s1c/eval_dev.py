"""S1c evaluation on DaLiA development folds S10–S12 (models trained LOSO on the other 14 subjects).
(1) E2 MAE: CNN argmax and CNN+HMM (HMM sigma selected leave-one-dev-subject-out), mean ± sd over seeds 0–4.
(2) K-b re-check: B0 (HR-PPG > 120) FAR vs D1 reference (pilot rules) and FAR_any.
(3) Injection validation: Delta = +20/+30/+40 bpm, plateau 180 s, onsets in ACC rest/light windows,
    seeds 0–4; recovered offset = median E2 HR on the plateau − median E2 HR in the 60 s before onset.
    uv run python 05_experiments/s1c/eval_dev.py
"""
import sys, json, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia, e2cnn, features, events, stats, inject
from h32.e2cnn import HR_GRID
torch.set_num_threads(8)
KIND = sys.argv[1] if len(sys.argv) > 1 else "fcn"          # "cnn" (v2.1 SpecCNN) or "fcn" (deviation #5)
PFX = "" if KIND == "cnn" else "fcn_"
GRID = HR_GRID if KIND == "cnn" else e2cnn.FCN_GRID
MODEL = e2cnn.SpecCNN if KIND == "cnn" else e2cnn.SpecFCN
TRACK = (lambda p, sg: e2cnn.hmm_track(p, sg)) if KIND == "cnn" else (lambda p, sg: e2cnn.hmm_track_grid(p, GRID, sg))
R = ROOT / "05_experiments/s1c/results"
DEV = ["S10", "S11", "S12"]; SEEDS = [0, 1, 2, 3, 4]; SIGMAS = [4.0, 8.0, 12.0, 16.0]
Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz")
y = {s: Z[f"{s}_y"] for s in DEV}
P = {(s, k): np.load(R / f"{PFX}proba_{s}_seed{k}.npy") for s in DEV for k in SEEDS}

mae_arg = {s: [float(np.mean(np.abs(GRID[P[(s, k)].argmax(1)] - y[s]))) for k in SEEDS] for s in DEV}
trk = {(s, k, sg): TRACK(P[(s, k)], sg) for s in DEV for k in SEEDS for sg in SIGMAS}
mae_trk = {(s, sg): float(np.mean([np.mean(np.abs(trk[(s, k, sg)] - y[s])) for k in SEEDS])) for s in DEV for sg in SIGMAS}
sig_sel = {s: min(SIGMAS, key=lambda sg: np.mean([mae_trk[(o, sg)] for o in DEV if o != s])) for s in DEV}
mae_e2 = {s: [float(np.mean(np.abs(trk[(s, k, sig_sel[s])] - y[s]))) for k in SEEDS] for s in DEV}

hours = {s: ((len(y[s]) - 1) * 2 + 8) / 3600 for s in DEV}
kb = []
for k in SEEDS:
    fa = det = nref = n_al = 0
    for s in DEV:
        hp = trk[(s, k, sig_sel[s])]
        ref = events.episodes(y[s] > 120, 3, 5); al = events.episodes(hp > 120, 3, 5)
        f, d = events.match(al, ref, 10); fa += f; det += d; nref += len(ref); n_al += len(al)
    H = sum(hours.values())
    kb.append(dict(seed=k, n_fa=fa, far=fa / H, ci=list(stats.poisson_rate_ci(fa, H)[1:]), det=det, n_ref=nref,
                   far_any=n_al / H))

inj = {}
for delta in (20, 30, 40):
    rec = []
    for s in DEV:
        d = io_dalia.load_subject(s); n = len(d["label"])
        lev = np.array([np.std(np.linalg.norm(d["acc"][i * 64:i * 64 + 256], axis=1)) for i in range(n)]) < 0.10
        for k in SEEDS:
            rng = np.random.default_rng(1000 * k + delta)
            onsets = inject.schedule(n, lev, rng, per_hour=3, min_gap_s=600, dur_s=240, start_s=120)
            m = MODEL(); m.load_state_dict(torch.load(R / f"{PFX}model_{s}_seed{k}.pt")); m.eval()
            ppg = d["bvp"].copy(); ref = d["label"].copy()
            for t0 in onsets:
                ppg, ref, _ = inject.inject(ppg, 64, ref, t0, delta, ramp_s=30, plateau_s=180)
            X = features.spectra(ppg, d["acc"], n)
            hp = TRACK(e2cnn.predict_proba(m, X), sig_sel[s])
            c = np.arange(len(hp)) * 2 + 4.0
            for t0 in onsets:
                pl = (c > t0 + 30) & (c < t0 + 210); pre = (c > t0 - 60) & (c < t0)
                ref_pl = (c > t0 + 30) & (c < t0 + 210)
                rec.append(dict(subject=s, seed=k, t0=t0, recovered=float(np.median(hp[pl]) - np.median(hp[pre])),
                                ref_rise=float(np.median(ref[ref_pl[:len(ref)]]) - np.median(ref[pre[:len(ref)]]))))
    r = np.array([x["recovered"] for x in rec])
    inj[delta] = dict(n_events=len(rec), recovered_median=float(np.median(r)), recovered_iqr=[float(np.percentile(r, 25)), float(np.percentile(r, 75))],
                      frac_ge_half=float(np.mean(r >= delta / 2)), events=rec)

out = dict(sigma_selected=sig_sel, mae_cnn_argmax=mae_arg, mae_e2=mae_e2, kb=kb, injection=inj,
           published=dict(source="Reiss 2019 Table 10/12", CNN_ensemble=dict(S10=4.03, S11=9.22, S12=9.35),
                          SpaMaPlus=dict(S10=6.17, S11=15.15, S12=12.03), CNN_constrained_all="9.99 ± 5.9 (26K params)"))
json.dump(out, open(R / f"eval_dev_{KIND}.json", "w"), indent=2)
print("sigma:", sig_sel)
for s in DEV:
    print(s, "CNN argmax MAE %.2f ± %.2f" % (np.mean(mae_arg[s]), np.std(mae_arg[s])), "| E2 (CNN+HMM) %.2f ± %.2f" % (np.mean(mae_e2[s]), np.std(mae_e2[s])))
for x in kb:
    print("K-b seed", x["seed"], "B0 FA=%d FAR=%.2f [%.2f,%.2f] det=%d/%d FAR_any=%.2f" % (x["n_fa"], x["far"], x["ci"][0], x["ci"][1], x["det"], x["n_ref"], x["far_any"]))
for dl, v in inj.items():
    print("inject +%d: n=%d recovered median %.1f IQR %s frac>=half %.2f" % (dl, v["n_events"], v["recovered_median"], [round(a, 1) for a in v["recovered_iqr"]], v["frac_ge_half"]))
