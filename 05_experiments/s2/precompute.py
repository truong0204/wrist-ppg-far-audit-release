"""S2 round-1 (Cowork review), DEV ONLY: cache E2 outputs for every analysis of 05_experiments/s2.
Recordings: WildPPG dev e61/u7y/ngh (zero-shot all-DaLiA SpecFCN) and DaLiA dev S10-S12 (LOSO fold models;
the other 12 DaLiA subjects have no out-of-subject E2 model). Confirmatory PPG stays locked.
Per (recording, seed 0-4): clean run + injected runs Delta = 20/30/40 (onsets among ALL windows, any state,
<= 3/h, >= 10 min apart, ramp 30 s, plateau 180 s; rng seed 1000 k + Delta + 7) + one calibration-contamination
run (Delta = 30 injected at the start of calibration block j = 2 of each state that has >= 6 complete blocks).
Stored: SQI and HMM tracks for sigma = 3, 5, 8. Output: 02_data/derived/s2_cache/*.npz (git-ignored).
    uv run python 05_experiments/s2/precompute.py
"""
import sys, time, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W, io_dalia, e2cnn, features, inject, method as M
torch.set_num_threads(8)
R1C = ROOT / "05_experiments/s1c/results"; OUT = ROOT / "02_data/derived/s2_cache"; OUT.mkdir(parents=True, exist_ok=True)
GRID = e2cnn.FCN_GRID; SIGMAS = (3.0, 5.0, 8.0); TH = 0.02896360121667385; SEEDS = range(5); t0 = time.time()


def recs():
    for s in W.DEV:
        d = W.load_wrist(s); p, a = features.to_dalia_rates(d["ppg"], d["fs_ppg"], d["acc"], d["fs_acc"])
        n = (len(p) // 64 - 8) // 2 + 1
        ref2 = np.repeat(d["hr_ref"], 4)[:n]; ref2 = np.concatenate([ref2, np.full(n - len(ref2), np.nan)])
        yield "WildPPG", s, p, a, n, ref2, lambda k: R1C / f"fcn_all_seed{k}.pt"
    for s in ("S10", "S11", "S12"):
        d = io_dalia.load_subject(s); n = len(d["label"])
        yield "DaLiA", s, d["bvp"], d["acc"], n, d["label"].astype(float), (lambda k, s=s: R1C / f"fcn_model_{s}_seed{k}.pt")


def run(m, ppg, acc, n):
    P = e2cnn.predict_proba(m, features.spectra(ppg, acc, n))
    return P.max(1).astype(np.float32), {sg: e2cnn.hmm_track_grid(P, GRID, sg).astype(np.float32) for sg in SIGMAS}


for src, sid, ppg, acc, n, ref2, mpath in recs():
    st = M.states(acc, n)
    for k in SEEDS:
        f = OUT / f"{src}_{sid}_seed{k}.npz"
        if f.exists():
            continue
        m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(mpath(k))); m.eval()
        sq, hp = run(m, ppg, acc, n); nn = len(sq)
        st_ = st[:nn]
        save = dict(st=st_, ref=ref2[:nn], sq=sq, **{f"hp{int(s)}": v for s, v in hp.items()})
        for delta in (20, 30, 40):
            rng = np.random.default_rng(1000 * k + delta + 7)
            on = inject.schedule(nn, np.ones(nn, bool), rng, per_hour=3, min_gap_s=600, dur_s=240, start_s=120)
            pp = ppg.copy(); rr = ref2.copy()
            for o in on:
                pp, rr, _ = inject.inject(pp, 64, rr, o, delta, ramp_s=30, plateau_s=180)
            sqi, hpi = run(m, pp, acc, n)
            save[f"on{delta}"] = np.array(on); save[f"sq{delta}"] = sqi
            for s, v in hpi.items(): save[f"hp{int(s)}_{delta}"] = v
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
        sqc, hpc = run(m, pp, acc, n)
        save["on_cal"] = np.array(on_c); save["sq_cal"] = sqc
        for s, v in hpc.items(): save[f"hp{int(s)}_cal"] = v
        np.savez_compressed(f, **save)
        print(src, sid, "seed", k, "n", nn, "cal events", len(on_c), "%.0f s" % (time.time() - t0), flush=True)
print("done %.0f s" % (time.time() - t0))
