"""Rev 5 D4: finite-sample bound of ACI (Gibbs & Candes 2021, NeurIPS, Proposition 4.1, p. 6; full text checked
2026-11-22, SHA256 09bafbdf...035a): with probability one, for all T,
    | (1/T) sum_t err_t - alpha | <= (max{alpha_1, 1 - alpha_1} + gamma) / (gamma T).
Our block ACI (b4_aci.py) satisfies its Lemma 4.1 premise (a_t < 0 -> tau = +inf -> err = 0; a_t > 1 -> tau = -inf).
Here err_t = 1{alarm in test block t}, so the bound on the block alarm fraction times 6 is a bound on alarms per
monitored hour (true events included). Computed for each dev cell with T = number of ACI test blocks (seed-averaged),
gamma = 0.05, alpha_1 = alpha* in {1/7, 1/3}. "Vacuous" = the bound allows every alarm fraction in [0, 1].
    uv run python 05_experiments/s2/r5_aci_bound.py
"""
import sys, json, pathlib, itertools, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"; TH = 0.02896360121667385; G = 0.05
STN = ["rest", "light", "vigorous"]; out = {}
for astar in (1 / 7, 1 / 3):
    rows = []
    for sid in ("e61", "u7y", "ngh"):
        for k in range(3):
            Ts = []
            for s in range(5):
                z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; mon = z["sq"] >= TH; blk = M.blocks(mon, st)
                Ts.append(max(M.complete_blocks(mon, st, blk)[k] - 6, 0))
            T = float(np.mean(Ts)); b = (max(astar, 1 - astar) + G) / (G * T) if T else np.inf
            lo, hi = max(astar - b, 0.0), min(astar + b, 1.0)
            rows.append(dict(sid=sid, state=STN[k], T=T, bound=b, alarm_frac_range=[lo, hi], far_range_per_h=[6 * lo, 6 * hi],
                             vacuous=bool(lo == 0.0 and hi == 1.0)))
    out[f"alpha_{astar:.4f}"] = rows
    print("alpha*", round(astar, 4)); [print("  ", r["sid"], r["state"], "T %.1f" % r["T"], "bound %.2f" % r["bound"], "FAR in [%.2f, %.2f]/h" % tuple(r["far_range_per_h"]), "VACUOUS" if r["vacuous"] else "") for r in rows]
json.dump(out, open(OUT / "r5_aci_bound.json", "w"), indent=1)
