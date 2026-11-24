"""Exploratory, WildPPG DEV only (e61, u7y, ngh), seed 0, zero-shot all-DaLiA SpecFCN.
Why do ~half of the injected events in u7y/ngh show no recovered rise (wildppg_dev_zeroshot.json)?
Per event (same onsets as wildppg_dev_check.py, seed 0): recovered offset from argmax and from the HMM at
sigma = 3 / 6 / 12 bpm, and from the raw PPG spectral peak in 40-200 bpm (no network); pre-onset E2 error vs reference; argmax spread before onset (a signal-quality proxy);
reference-free SQI candidate = median network max-probability in the 60 s before onset (PPG/ACC only).
Not a confirmatory analysis; any choice it motivates goes into prereg v2.2 as a dev-based deviation.
    uv run python 05_experiments/s1c/wildppg_inject_diag.py
"""
import sys, json, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W, e2cnn, features, inject
torch.set_num_threads(4)
R = ROOT / "05_experiments/s1c/results"; GRID = e2cnn.FCN_GRID; K = 0; SIGMAS = (3.0, 6.0, 12.0)
m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(R / f"fcn_all_seed{K}.pt")); m.eval()
out = []; SQ = {}
for s in W.DEV:
    d = W.load_wrist(s)
    ppg64, acc32 = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
    n = (len(ppg64) // 64 - 8) // 2 + 1
    lev = np.array([np.std(np.linalg.norm(acc32[i * 64:i * 64 + 256], axis=1)) for i in range(n)]) < 0.10
    ref2 = np.repeat(d["hr_ref"], 4)[:n]; ref2 = np.concatenate([ref2, np.full(n - len(ref2), np.nan)])
    for delta in (20, 30, 40):
        rng = np.random.default_rng(1000 * K + delta)
        onsets = inject.schedule(n, lev, rng, per_hour=3, min_gap_s=600, dur_s=240, start_s=120)
        ppg = ppg64.copy(); r = ref2.copy()
        for t0 in onsets:
            ppg, r, _ = inject.inject(ppg, 64, r, t0, delta, ramp_s=30, plateau_s=180)
        XS = features.spectra(ppg, acc32, n)
        P = e2cnn.predict_proba(m, XS); arg = GRID[P.argmax(1)]
        band = (features.FQ * 60 >= 40) & (features.FQ * 60 <= 200)
        raw = features.FQ[band][XS[:, 0, band].argmax(1)] * 60
        H = {sg: e2cnn.hmm_track_grid(P, GRID, sg) for sg in SIGMAS}
        c = np.arange(n) * 2 + 4.0
        conf = P.max(1)
        if delta == 20: SQ[s] = conf
        for t0 in onsets:
            pl = (c > t0 + 30) & (c < t0 + 210); pre = (c > t0 - 60) & (c < t0)
            out.append(dict(subject=s, delta=delta, t0=t0, ref_pre=float(np.nanmedian(ref2[pre])),
                            rec_argmax=float(np.median(arg[pl]) - np.median(arg[pre])),
                            rec_rawpeak=float(np.median(raw[pl]) - np.median(raw[pre])),
                            **{f"rec_hmm{int(sg)}": float(np.median(H[sg][pl]) - np.median(H[sg][pre])) for sg in SIGMAS},
                            pre_err_hmm3=float(abs(np.median(H[3.0][pre]) - np.nanmedian(ref2[pre]))),
                            sqi_pre=float(np.median(conf[pre])),
                            pre_argmax_iqr=float(np.subtract(*np.percentile(arg[pre], [75, 25])))))
    print(s, "done", flush=True)
json.dump(out, open(R / "wildppg_inject_diag.json", "w"), indent=1)
E = out
for s in W.DEV:
    for dl in (20, 30, 40):
        x = [e for e in E if e["subject"] == s and e["delta"] == dl]
        f = lambda k: np.median([e[k] for e in x])
        print(s, dl, len(x), "raw %.1f argmax %.1f hmm3 %.1f hmm6 %.1f hmm12 %.1f | pre_err %.1f pre_iqr %.1f ref_pre %.0f" % (
            f("rec_rawpeak"), f("rec_argmax"), f("rec_hmm3"), f("rec_hmm6"), f("rec_hmm12"), f("pre_err_hmm3"), f("pre_argmax_iqr"), f("ref_pre")))
ok = np.array([e["pre_err_hmm3"] < 5 for e in E]); hit = np.array([e["rec_hmm3"] >= e["delta"] / 2 for e in E])
print("recovered>=half | pre_err<5: %.2f (n=%d); | pre_err>=5: %.2f (n=%d)" % (hit[ok].mean(), ok.sum(), hit[~ok].mean(), (~ok).sum()))
allc = np.concatenate(list(SQ.values()))
for q in (0.0, 0.25, 0.5):
    th = float(np.quantile(allc, q)); keep = np.array([e["sqi_pre"] >= th for e in E])
    for dl in (20, 30, 40):
        x = np.array([e["rec_hmm3"] for e, k in zip(E, keep) if k and e["delta"] == dl])
        print("SQI>=q%.2f (%.3f): windows kept %.2f | +%d n=%d median %.1f frac>=half %.2f" % (
            q, th, np.mean(allc >= th), dl, len(x), np.median(x), np.mean(x >= dl / 2)))
    print("  kept per ID:", {s_: round(float(np.mean(SQ[s_] >= th)), 2) for s_ in SQ})
