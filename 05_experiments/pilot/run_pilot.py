"""T3 pilot: B0 (hard 120 bpm on HR-PPG) vs B2 (population split conformal, LOSO) on 3 DaLiA subjects.

    uv run python 05_experiments/pilot/run_pilot.py
Deterministic (no random step): the >=5-seed rule does not apply; recorded in the output.
"""
from __future__ import annotations
import json, pathlib, platform, sys, datetime, subprocess
import numpy as np, scipy, yaml

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia, e1, events, stats  # noqa: E402

ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling", 5: "driving",
       6: "lunch", 7: "walking", 8: "working"}


def prepare(sid):
    d = io_dalia.load_subject(sid)
    n = len(d["label"])
    hr_ppg, motion = e1.estimate(d["bvp"], d["acc"], n)
    act = io_dalia.window_activity(d["activity"], n)
    return dict(sid=sid, n=n, hours=((n - 1) * 2 + 8) / 3600.0, hr_ecg=d["label"], hr_ppg=hr_ppg,
                motion=motion, act=act)


def evaluate(s, alarm_flag, ev):
    ref_eps = events.episodes(s["hr_ecg"] > ev["thr"], ev["k"], ev["gap"])
    al_eps = events.episodes(alarm_flag, ev["k"], ev["gap"])
    n_fa, n_det = events.match(al_eps, ref_eps, ev["tol"])
    # activity of each false-alarm episode (majority over its windows)
    fa_act = {}
    R = [(a - ev["tol"], b + ev["tol"]) for a, b in (events.to_seconds(r) for r in ref_eps)]
    for ep in al_eps:
        t = events.to_seconds(ep)
        if not any(t[0] < r[1] and r[0] < t[1] for r in R):
            a = ACT[int(np.bincount(s["act"][ep[0]:ep[1] + 1]).argmax())]
            fa_act[a] = fa_act.get(a, 0) + 1
    neg = s["hr_ecg"] <= ev["thr"]
    return dict(n_ref=len(ref_eps), n_alarm=len(al_eps), n_fa=n_fa, n_detected=n_det,
                window_fpr=float(np.mean(alarm_flag[neg])) if neg.any() else None, fa_by_activity=fa_act)


def main():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    thr = cfg["threshold_bpm"]
    ev = dict(thr=thr, k=cfg["event"]["k_min_windows"], gap=cfg["event"]["merge_gap_windows"],
              tol=cfg["event"]["match_tolerance_s"])
    S = {sid: prepare(sid) for sid in cfg["subjects"]}
    rows = []
    for sid, s in S.items():
        err = s["hr_ppg"] - s["hr_ecg"]
        base = dict(subject=sid, hours=s["hours"], windows=s["n"], e1_mae_bpm=float(np.nanmean(np.abs(err))),
                    motion_frac=float(s["motion"].mean()))
        rows.append(dict(base, method="B0", q_bpm=0.0, **evaluate(s, s["hr_ppg"] > thr, ev)))
        cal = np.concatenate([S[o]["hr_ppg"] - S[o]["hr_ecg"] for o in S if o != sid])  # LOSO: other subjects only
        for a in cfg["b2"]["alphas"]:
            q = stats.conformal_quantile(cal, a)
            rows.append(dict(base, method=f"B2(a={a})", q_bpm=q, **evaluate(s, s["hr_ppg"] - q > thr, ev)))
    # pooled
    pooled = []
    for m in sorted({r["method"] for r in rows}, key=lambda x: (x != "B0", x)):
        R = [r for r in rows if r["method"] == m]
        H = sum(r["hours"] for r in R); fa = sum(r["n_fa"] for r in R)
        nref = sum(r["n_ref"] for r in R); ndet = sum(r["n_detected"] for r in R)
        rate, lo, hi = stats.poisson_rate_ci(fa, H)
        per_subj = [r["n_fa"] / r["hours"] for r in R]
        pooled.append(dict(method=m, subjects=len(R), hours=H, n_fa=fa, far_per_h=rate, far_ci95=[lo, hi],
                           far_per_subject=per_subj, n_ref=nref, n_detected=ndet,
                           sensitivity=(ndet / nref) if nref else None))
    b0 = next(p for p in pooled if p["method"] == "B0")
    out = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        env=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                 platform=platform.platform(), machine=platform.machine()),
        seeds="not applicable: pipeline has no random step", config=cfg, e1_params=e1.PARAMS,
        per_subject=rows, pooled=pooled,
        gate_g1_part_far=dict(b0_far_per_h=b0["far_per_h"], stop=b0["far_per_h"] < cfg["gate_g1"]["far_b0_min_per_hour"]),
    )
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "pilot_results.json", "w"), indent=2, default=float)
    print(json.dumps(pooled, indent=1, default=float))
    print("G1 (FAR part) stop:", out["gate_g1_part_far"])


if __name__ == "__main__":
    main()
