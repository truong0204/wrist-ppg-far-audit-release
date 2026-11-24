"""G1 second condition: number of reference episodes (HR-ECG only, no estimator, no alarms) on every
available DaLiA subject, for the primary and the alternative event definitions in config.yaml.

    uv run python 05_experiments/pilot/count_ref_episodes.py
Label-only statistic (like class prevalence): it uses no PPG, fits nothing, and selects nothing.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np, yaml

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia, events  # noqa: E402

cfg = yaml.safe_load(open(HERE / "config.yaml"))
ev = cfg["event"]
subs = list(io_dalia.DEV) + list(io_dalia.CONFIRMATORY)   # labels + ACC only (need_ppg=False)
subs = sorted(subs, key=lambda s: int(s[1:]))
res = {"subjects": subs, "definitions": []}
for d in cfg["alternative_events_label_only"]:
    per, hours = {}, 0.0
    for sid in subs:
        z = io_dalia.load_subject(sid, need_ppg=False)
        hr = z["label"]; n = len(hr); hours += ((n - 1) * 2 + 8) / 3600
        if "threshold_bpm" in d:
            thr = d["threshold_bpm"]
        else:
            act = io_dalia.window_activity(z["activity"], n)
            thr = float(np.median(hr[act == 1])) + d["relative_to_sitting_median_bpm"]
        per[sid] = len(events.episodes(hr > thr, ev["k_min_windows"], ev["merge_gap_windows"]))
    res["definitions"].append(dict(name=d["name"], total=sum(per.values()), per_subject=per,
                                   subjects_with_zero=sum(v == 0 for v in per.values()), hours=hours))
(HERE / "results").mkdir(exist_ok=True)
json.dump(res, open(HERE / "results" / f"ref_episode_counts_{len(subs)}subj.json", "w"), indent=2)
for r in res["definitions"]:
    print(f"{r['name']:<10} total={r['total']:>4}  zero-subjects={r['subjects_with_zero']}  hours={r['hours']:.1f}  {r['per_subject']}")
