"""S1b task B: reference-episode counts on all 15 DaLiA subjects using ONLY HR-ECG labels and wrist ACC.
No PPG is loaded (need_ppg=False; confirmatory PPG is locked in io_dalia).

    uv run python 05_experiments/s1b/B_counts.py
Motion-intensity levels: prereg_v2 §3. Episode rules: prereg v1 P4 (k>=3 windows, merge gap <=5).
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia, events  # noqa: E402

K, GAP = 3, 5
REST, VIG = 0.02, 0.10                       # g, prereg_v2 §3
WIN_PER_MIN = 30                             # 2 s shift
CAL_MIN = 60                                 # D3b: expected HR learned on the first 60 min only
BUFFER_W = 4                                 # 8 s buffer between learning and counting parts


def motion_levels(acc, n):
    s = np.array([np.std(np.linalg.norm(acc[i * 64:i * 64 + 256], axis=1)) for i in range(n)])
    return np.where(s < REST, 0, np.where(s < VIG, 1, 2)), s


def run_length(flag):
    out = np.zeros(len(flag), int); c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0; out[i] = c
    return out


def count(flag):
    return len(events.episodes(flag, K, GAP))


def main():
    subs = sorted(io_dalia.DEV + io_dalia.CONFIRMATORY, key=lambda s: int(s[1:]))
    per = {}
    for sid in subs:
        z = io_dalia.load_subject(sid, need_ppg=False)
        hr = z["label"]; n = len(hr); hours = ((n - 1) * 2 + 8) / 3600
        lev, s = motion_levels(z["acc"], n)
        rr = run_length(lev == 0)
        r = dict(hours=hours, level_frac=[float(np.mean(lev == k)) for k in range(3)],
                 hr_rest_p99=float(np.percentile(hr[lev == 0], 99)) if (lev == 0).any() else None)
        r["D1_HR>120"] = count(hr > 120)
        for T in (1, 3, 5):
            r[f"D2_T{T}"] = count((hr > 100) & (rr >= T * WIN_PER_MIN))
        r["D2_rest_minutes_total"] = float(np.sum(lev == 0) / WIN_PER_MIN)
        # D3a (circular, in-sample): HR above the subject's own 99th percentile within the same motion level
        q = {k: np.percentile(hr[lev == k], 99) for k in range(3) if (lev == k).sum() >= WIN_PER_MIN}
        r["D3a_insample_q99"] = count(np.array([lev[i] in q and hr[i] > q[lev[i]] for i in range(n)]))
        # D3b (chronological): expected HR per level from the first CAL_MIN minutes; count only afterwards
        c = CAL_MIN * WIN_PER_MIN
        if n > c + BUFFER_W:
            hc, lc = hr[:c], lev[:c]
            allq = np.percentile(hc, 99)
            qb = {k: (np.percentile(hc[lc == k], 99) if (lc == k).sum() >= WIN_PER_MIN else allq) for k in range(3)}
            ht, lt = hr[c + BUFFER_W:], lev[c + BUFFER_W:]
            r["D3b_hours_counted"] = (len(ht) * 2) / 3600
            r["D3b_q99"] = count(np.array([ht[i] > qb[lt[i]] for i in range(len(ht))]))
            r["D3b_q99_plus10"] = count(np.array([ht[i] > qb[lt[i]] + 10 for i in range(len(ht))]))
            r["D3b_levels_seen_in_first_60min"] = [int((lc == k).sum() >= WIN_PER_MIN) for k in range(3)]
        per[sid] = r
    keys = ["D1_HR>120", "D2_T1", "D2_T3", "D2_T5", "D3a_insample_q99", "D3b_q99", "D3b_q99_plus10"]
    summary = {}
    for k in keys:
        for grp, ids in (("all15", subs), ("confirm12", io_dalia.CONFIRMATORY), ("dev3", io_dalia.DEV)):
            v = [per[s].get(k, 0) for s in ids]
            summary[f"{k}|{grp}"] = dict(total=int(sum(v)), subjects_ge1=int(sum(x >= 1 for x in v)), n_subjects=len(ids))
    out = dict(per_subject=per, summary=summary, hours_all=sum(p["hours"] for p in per.values()),
               hours_confirm=sum(per[s]["hours"] for s in io_dalia.CONFIRMATORY),
               d3b_hours_confirm=sum(per[s].get("D3b_hours_counted", 0) for s in io_dalia.CONFIRMATORY),
               params=dict(K=K, GAP=GAP, REST=REST, VIG=VIG, CAL_MIN=CAL_MIN, BUFFER_W=BUFFER_W))
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "B_counts.json", "w"), indent=2, default=float)
    print("hours all=%.1f confirm=%.1f  D3b counted hours (confirm)=%.1f" % (out["hours_all"], out["hours_confirm"], out["d3b_hours_confirm"]))
    print("subject  hrs  rest/light/vig   D1  D2T1 D2T3 D2T5  D3a D3b D3b+10  HRrest_p99")
    for s in subs:
        p = per[s]
        print(f"{s:<5} {p['hours']:5.2f}  {p['level_frac'][0]:.2f}/{p['level_frac'][1]:.2f}/{p['level_frac'][2]:.2f}  "
              f"{p['D1_HR>120']:4d} {p['D2_T1']:4d} {p['D2_T3']:4d} {p['D2_T5']:4d} {p['D3a_insample_q99']:4d} "
              f"{p.get('D3b_q99', '-'):>4} {p.get('D3b_q99_plus10', '-'):>5}  {p['hr_rest_p99']:.0f}")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
