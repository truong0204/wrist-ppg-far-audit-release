"""S2 r1 m7 (Cowork): detection delay vs onset speed. DEV ONLY (WildPPG e61/u7y/ngh, DaLiA S10-S12).
Extra injected runs: Delta = 30, ramp 10 s and 60 s (plateau 180 s); onsets among all windows, <= 3/h,
>= 10 min apart; rng seed 1000 k + 37 + ramp. HMM sigma = 3 / 5 / 8. (Ramp 30 s is in precompute.py.)
Output: 02_data/derived/s2_cache/ramp_<src>_<sid>_seed<k>.npz
    uv run python 05_experiments/s2/precompute_ramp.py
"""
import sys, time, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W, io_dalia, e2cnn, features, inject
torch.set_num_threads(4)
R1C = ROOT / "05_experiments/s1c/results"; OUT = ROOT / "02_data/derived/s2_cache"; OUT.mkdir(parents=True, exist_ok=True)
GRID = e2cnn.FCN_GRID; SIGMAS = (3.0, 5.0, 8.0); t0 = time.time()


def recs():
    for s in W.DEV:
        d = W.load_wrist(s); p, a = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
        n = (len(p) // 64 - 8) // 2 + 1
        ref2 = np.repeat(d["hr_ref"], 4)[:n]; ref2 = np.concatenate([ref2, np.full(n - len(ref2), np.nan)])
        yield "WildPPG", s, p, a, n, ref2, lambda k: R1C / f"fcn_all_seed{k}.pt"
    for s in ("S10", "S11", "S12"):
        d = io_dalia.load_subject(s); n = len(d["label"])
        yield "DaLiA", s, d["bvp"], d["acc"], n, d["label"].astype(float), (lambda k, s=s: R1C / f"fcn_model_{s}_seed{k}.pt")


for src, sid, ppg, acc, n, ref2, mpath in recs():
    for k in range(5):
        f = OUT / f"ramp_{src}_{sid}_seed{k}.npz"
        if f.exists():
            continue
        m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(mpath(k))); m.eval()
        save = {}
        for ramp in (10, 60):
            rng = np.random.default_rng(1000 * k + 37 + ramp)
            on = inject.schedule(n, np.ones(n, bool), rng, per_hour=3, min_gap_s=600, dur_s=2 * ramp + 180 + 60, start_s=120)
            pp = ppg.copy(); rr = ref2.copy()
            for o in on:
                pp, rr, _ = inject.inject(pp, 64, rr, o, 30, ramp_s=ramp, plateau_s=180)
            P = e2cnn.predict_proba(m, features.spectra(pp, acc, n))
            save[f"on_r{ramp}"] = np.array(on); save[f"sq_r{ramp}"] = P.max(1).astype(np.float32)
            for sg in SIGMAS:
                save[f"hp{int(sg)}_r{ramp}"] = e2cnn.hmm_track_grid(P, GRID, sg).astype(np.float32)
        np.savez_compressed(f, **save)
        print(src, sid, "seed", k, "%.0f s" % (time.time() - t0), flush=True)
print("done %.0f s" % (time.time() - t0))
