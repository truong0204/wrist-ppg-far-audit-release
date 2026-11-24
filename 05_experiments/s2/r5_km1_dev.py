"""Rev 5 item 1: K with m = 1 (prereg §2 sensitivity analysis: thinned calibration from the first n_cal = 12 blocks,
tau = max of blocks 0, 2, ..., 10; buffer 1 -> test blocks >= 13), on dev; next to K (m = 0) on its own test blocks
and on the same blocks >= 13. Seed-averaged; per cell and per person. X = #cells with rate > 2/h.
    uv run python 05_experiments/s2/r5_km1_dev.py
"""
import sys, json, pathlib, itertools, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"; TH = 0.02896360121667385
STN = ["rest", "light", "vigorous"]; cells = {}
for sid, s in itertools.product(("e61", "u7y", "ngh"), range(5)):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; mon = z["sq"] >= TH; blk = M.blocks(mon, st)
    D = M.debounce(z["hp3"], mon, st, blk); Sb = M.block_scores(D, st, blk); nc = M.complete_blocks(mon, st, blk)
    for k in range(3):
        S = Sb[k][:nc[k]]
        if len(S) < 14: continue
        t0, t1 = S[:6].max(), S[0:12:2].max()
        cells.setdefault((sid, STN[k]), []).append(dict(K=6 * np.sum(S[6:] > t0) / (len(S) - 6), K13=6 * np.sum(S[13:] > t0) / (len(S) - 13),
                                                        Km1=6 * np.sum(S[13:] > t1) / (len(S) - 13), T13=len(S) - 13))
rows = [dict(sid=a, state=b, T_m1=float(np.mean([x["T13"] for x in v])), **{m: float(np.mean([x[m] for x in v])) for m in ("K", "K13", "Km1")},
             seeds_above_r={m: int(sum(x[m] > 2 for x in v)) for m in ("K", "K13", "Km1")}) for (a, b), v in cells.items()]
Xbar = {m: float(np.mean([sum(c[m] > 2 for c in [cells[key][s] for key in cells]) for s in range(5)])) for m in ("K", "K13", "Km1")}
out = dict(cells=rows, Xbar=Xbar, n_cells=len(rows)); json.dump(out, open(OUT / "r5_km1_dev.json", "w"), indent=1)
for r in rows: print(r)
print("Xbar (seed-avg #cells > 2/h):", Xbar, "of", len(rows), "cells with >= 14 blocks")
