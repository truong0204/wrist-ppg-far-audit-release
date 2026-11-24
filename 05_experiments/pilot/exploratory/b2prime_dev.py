"""EXPLORATORY (not pre-registered): B2' = population split-conformal threshold on HR-PPG computed
from NEGATIVE windows (HR-ECG <= 120) of the other development subjects; alarm if HR-PPG > tau.
Targets per-window false-positive probability <= alpha. Development subjects S10-S12 only.
Purpose: check whether a fairer population baseline than residual-based B2 is non-degenerate
before Part M of the prereg is frozen. Same E1, same event rules as the pilot.
"""
import json, pathlib, sys
import numpy as np, yaml
HERE = pathlib.Path(__file__).parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "04_code")); sys.path.insert(0, str(HERE.parent))
from h32 import stats, events  # noqa
import run_pilot as rp  # reuse prepare/evaluate exactly
cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); thr = cfg["threshold_bpm"]
ev = dict(thr=thr, k=cfg["event"]["k_min_windows"], gap=cfg["event"]["merge_gap_windows"], tol=cfg["event"]["match_tolerance_s"])
S = {s: rp.prepare(s) for s in cfg["subjects"]}
out = []
for a in (0.05, 0.01, 0.002):
    fa = det = nref = 0; H = 0; taus = {}
    for sid, s in S.items():
        cal = np.concatenate([S[o]["hr_ppg"][S[o]["hr_ecg"] <= thr] for o in S if o != sid])
        tau = stats.conformal_quantile(cal, a); taus[sid] = tau
        r = rp.evaluate(s, s["hr_ppg"] > tau, ev); fa += r["n_fa"]; det += r["n_detected"]; nref += r["n_ref"]; H += s["hours"]
    rate, lo, hi = stats.poisson_rate_ci(fa, H)
    out.append(dict(alpha=a, tau_bpm=taus, n_fa=fa, hours=H, far_per_h=rate, ci95=[lo, hi], detected=det, n_ref=nref))
    print(f"alpha={a}: tau={ {k: round(v,1) for k,v in taus.items()} } FA={fa} FAR={rate:.2f}/h [{lo:.2f},{hi:.2f}] det={det}/{nref}")
json.dump(out, open(HERE / "b2prime_dev.json", "w"), indent=2)
