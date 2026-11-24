"""S1b C4: vacuity table for BLOCK calibration (unit = block of L min, score = block max, refractory >= L).

Target r false alarms/hour -> blocks/hour = 60/L -> per-block alpha = r*L/60.
Finite split-conformal threshold needs n >= ceil((1-alpha)/alpha) calibration blocks (event-free).
Dependence: lag-1 autocorrelation rho of block maxima measured on DEV subjects (C_dev.json, E2 HR-PPG);
AR(1) effective-sample heuristic n_eff = n (1-rho)/(1+rho) -> inflate n by (1+rho)/(1-rho). This is a
heuristic, not a bound.
Test time left per person: DaLiA from HR-ECG label lengths + D1 event-free blocks (label-only, allowed on
confirmatory subjects); WildPPG from the reference-HR series length of the HF release (8 s windows).

    uv run python 05_experiments/s1b/C4_vacuity.py
"""
from __future__ import annotations
import json, math, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia  # noqa: E402

cdev = json.load(open(HERE / "results" / "C_dev.json"))
rho = {L: float(np.mean(cdev["autocorr_lag1"][f"L{L}|HR-PPG(E2)"])) for L in (1, 2, 5, 10)}

dal = {}
for sid in io_dalia.DEV + io_dalia.CONFIRMATORY:
    hr = io_dalia.load_subject(sid, need_ppg=False)["label"]
    dal[sid] = hr
wild_hours = []
try:
    import scipy.io as sio
    m = sio.loadmat(ROOT / "02_data/raw/wildppg_hf/WildPPG.mat", variable_names=["data_bpm_values"])
    wild_hours = [np.asarray(m["data_bpm_values"][i, 0]).size * 8 / 3600 for i in range(16)]
except FileNotFoundError:
    pass

rows = []
for L in (1, 2, 5, 10):
    w = 30 * L
    for r in (0.5, 1, 2):
        a = r * L / 60
        n_nom = math.ceil((1 - a) / a)
        infl = (1 + rho[L]) / (1 - rho[L])
        n_ess = math.ceil(n_nom * infl)
        res = dict(L_min=L, target_per_h=r, alpha_block=a, n_blocks_nominal=n_nom, cal_min_nominal=n_nom * L,
                   rho_dev_mean=rho[L], n_blocks_ess=n_ess, cal_min_ess=n_ess * L)
        for tag, need in (("nominal", n_nom), ("ess", n_ess)):
            left = {}
            for sid, hr in dal.items():
                nb = len(hr) // w; got = 0; t_end = None
                for b in range(nb):
                    if not (hr[b * w:(b + 1) * w] > 120).any():
                        got += 1
                    if got >= need:
                        t_end = (b + 1) * L; break
                total_min = len(hr) * 2 / 60
                left[sid] = 0.0 if t_end is None else max(0.0, (total_min - t_end - L) / 60)
            conf = [left[s] for s in io_dalia.CONFIRMATORY]
            res[f"dalia_test_h_{tag}"] = dict(median_conf=float(np.median(conf)), min_conf=float(np.min(conf)),
                                              n_conf_ge_0p5h=int(sum(x >= 0.5 for x in conf)),
                                              n_conf_ge_1h=int(sum(x >= 1 for x in conf)), per_subject=left)
            if wild_hours:
                cal_h = (n_nom if tag == "nominal" else n_ess) * L / 60
                wl = [max(0.0, h - cal_h - L / 60) for h in wild_hours]
                res[f"wild_test_h_{tag}"] = dict(median=float(np.median(wl)), min=float(np.min(wl)))
        rows.append(res)
(HERE / "results").mkdir(exist_ok=True)
json.dump(dict(rho=rho, rows=rows, wild_hours_per_subject=wild_hours), open(HERE / "results" / "C4_vacuity.json", "w"), indent=2)
print("rho(dev, E2 block max):", {k: round(v, 2) for k, v in rho.items()})
print("L  r    alpha   n_nom cal_nom | n_ess cal_ess | DaLiA test h (conf12) nominal: med/min/#>=0.5h | ess: med/min/#>=0.5h | Wild test h nom/ess med")
for x in rows:
    dn, de = x["dalia_test_h_nominal"], x["dalia_test_h_ess"]
    wn = x.get("wild_test_h_nominal", {}).get("median", float("nan")); we = x.get("wild_test_h_ess", {}).get("median", float("nan"))
    print(f"{x['L_min']:<2} {x['target_per_h']:<4} {x['alpha_block']:.4f} {x['n_blocks_nominal']:5d} {x['cal_min_nominal']:6d}  | {x['n_blocks_ess']:5d} {x['cal_min_ess']:6d}  | "
          f"{dn['median_conf']:.2f}/{dn['min_conf']:.2f}/{dn['n_conf_ge_0p5h']:2d}  | {de['median_conf']:.2f}/{de['min_conf']:.2f}/{de['n_conf_ge_0p5h']:2d} | {wn:.1f}/{we:.1f}")
