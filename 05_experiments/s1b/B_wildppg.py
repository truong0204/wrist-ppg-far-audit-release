"""S1b B (WildPPG part): reference-episode counts from the authors' cleaned HR (HF release WildPPG.mat,
variable data_bpm_values; 8 s windows; 0 = removed by cleaning). Label-only: no PPG variable is read
from the HF file (scipy loadmat variable_names restricts loading).

The HF file has no subject IDs. Index -> ID mapping is by recording length: for each per-ID polybox file
the 'sternum' struct is loaded only to read len(ecg.v); fs = 128 Hz; n_windows = len/(128*8). The struct
also holds sternum PPG arrays, which are discarded unread (documented in prereg_v2 change log).

    uv run python 05_experiments/s1b/B_wildppg.py
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np, scipy.io as sio

HERE = pathlib.Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "04_code"))
from h32 import events  # noqa: E402

DEV = ["e61", "u7y", "ngh"]
IDS = ["e61", "u7y", "ngh", "an0", "fex", "k2s", "kjd", "l38", "n31", "p5d", "p9p", "qm9", "ssx", "trh", "tz8", "w4p"]
K, GAP = 2, 1            # 8 s windows: >= 16 s, merge gaps <= 8 s (closest to DaLiA's 12 s / 10 s)

hf = sio.loadmat(ROOT / "02_data/raw/wildppg_hf/WildPPG.mat", variable_names=["data_bpm_values"])
bpm = [np.asarray(hf["data_bpm_values"][i, 0]).ravel().astype(float) for i in range(16)]
hf_len = [b.size for b in bpm]

id_len = {}
for sid in IDS:
    f = ROOT / "02_data/raw/wildppg" / f"WildPPG_Part_{sid}.mat"
    st = sio.loadmat(f, variable_names=["sternum"], squeeze_me=True, struct_as_record=False)["sternum"]
    id_len[sid] = int(np.asarray(st.ecg.v).size // (128 * 8)); del st
mapping = {}
for sid, L in id_len.items():
    cand = [i for i, h in enumerate(hf_len) if abs(h - L) <= 1]
    mapping[sid] = cand
ok = all(len(c) == 1 for c in mapping.values()) and len({c[0] for c in mapping.values()}) == 16

res = dict(id_n_windows=id_len, hf_n_windows=hf_len, mapping=mapping, mapping_unique=ok, per_subject={})
for sid in IDS:
    if len(mapping[sid]) != 1:
        continue
    b = bpm[mapping[sid][0]]; valid = b > 0
    r = dict(hf_index=mapping[sid][0], hours_total=b.size * 8 / 3600, hours_valid=float(valid.sum() * 8 / 3600),
             frac_removed=float(1 - valid.mean()))
    for thr in (100, 110, 120):
        r[f"HR>{thr}"] = len(events.episodes(valid & (b > thr), K, GAP))
    res["per_subject"][sid] = r
summ = {}
for grp, ids in (("confirm13", [s for s in IDS if s not in DEV]), ("dev3", DEV), ("all16", IDS)):
    for thr in (100, 110, 120):
        v = [res["per_subject"][s][f"HR>{thr}"] for s in ids if s in res["per_subject"]]
        summ[f"HR>{thr}|{grp}"] = dict(total=int(sum(v)), subjects_ge1=int(sum(x >= 1 for x in v)), n=len(v))
    summ[f"hours_valid|{grp}"] = float(sum(res["per_subject"][s]["hours_valid"] for s in ids if s in res["per_subject"]))
res["summary"] = summ
(HERE / "results").mkdir(exist_ok=True)
json.dump(res, open(HERE / "results" / "B_wildppg.json", "w"), indent=2)
print("mapping unique:", ok, {s: c for s, c in mapping.items()})
for s in IDS:
    if s in res["per_subject"]:
        p = res["per_subject"][s]
        print(f"{s} idx={p['hf_index']:2d} h={p['hours_total']:.1f} valid={p['hours_valid']:.1f} removed={p['frac_removed']:.2f}  >100:{p['HR>100']:3d} >110:{p['HR>110']:3d} >120:{p['HR>120']:3d}")
for k, v in summ.items():
    print(k, v)
