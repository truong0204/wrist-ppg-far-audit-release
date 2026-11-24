"""Regression check of the confirmatory pipeline on the 3 dev IDs (PI step 2).
(1) numbers recorded in prereg v2.2 (rev 6/7, §5 and §4) vs the new script (runs/<label>/results.json);
    tolerance = half a unit of the last digit printed in the prereg; anything outside is listed as a MISMATCH with
    its explanation (never "fixed" by editing the prereg number);
(2) E2 arrays of the new cache vs the dev cache 02_data/derived/s2_cache (byte-identical?);
(3) run 1 vs run 2: results.json byte-identical, caches byte-identical.
    uv run python 05_experiments/confirm/regression_check.py dryrun_dev_r8_1 dryrun_dev_r8_2
"""
import sys, json, pathlib, hashlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
L1, L2 = sys.argv[1], sys.argv[2]
R = json.loads((ROOT / f"05_experiments/confirm/runs/{L1}/results.json").read_text())
ids = R["ids"]; S = R["sensitivity"]["summary"]


def g(path):
    o = R
    for p in path: o = o[p]
    return o


# (prereg value, decimals, getter, where in the prereg, note)
CHECKS = [
    ("FAR K pooled", 1.40, 2, lambda: g(["FAR", "K", "pooled"]), "§5 FAR table"),
    ("FAR K CI low", 0.49, 2, lambda: g(["FAR", "K", "cluster_ci95"])[0], "§5 FAR table"),
    ("FAR K CI high", 2.16, 2, lambda: g(["FAR", "K", "cluster_ci95"])[1], "§5 FAR table"),
    ("FAR R pooled", 1.05, 2, lambda: g(["FAR", "R", "pooled"]), "§5 FAR table"),
    ("FAR R e61", 1.05, 2, lambda: g(["FAR", "R", "per_person", "e61"]), "§4.3 prediction"),
    ("FAR R u7y", 0.77, 2, lambda: g(["FAR", "R", "per_person", "u7y"]), "§4.3 prediction"),
    ("FAR R ngh", 1.24, 2, lambda: g(["FAR", "R", "per_person", "ngh"]), "§4.3 prediction"),
    ("FAR R-thin e61", 0.68, 2, lambda: g(["FAR", "Rthin", "per_person", "e61"]), "§5 rev 3"),
    ("FAR R-thin u7y", 0.80, 2, lambda: g(["FAR", "Rthin", "per_person", "u7y"]), "§5 rev 3"),
    ("FAR R-thin ngh", 1.16, 2, lambda: g(["FAR", "Rthin", "per_person", "ngh"]), "§5 rev 3"),
    ("FAR R-sel pooled", 2.71, 2, lambda: g(["FAR", "Rsel", "pooled"]), "§5 FAR table"),
    ("FAR ACI mean of persons", 0.72, 2, lambda: g(["FAR", "ACI", "mean_of_persons"]), "§5 B4 table"),
    ("FAR ACI e61", 0.64, 2, lambda: g(["FAR", "ACI", "per_person", "e61"]), "§5 B4 table"),
    ("FAR ACI u7y", 0.75, 2, lambda: g(["FAR", "ACI", "per_person", "u7y"]), "§5 B4 table"),
    ("FAR ACI ngh", 0.77, 2, lambda: g(["FAR", "ACI", "per_person", "ngh"]), "§5 B4 table"),
    ("X2 sum d", 1.57, 2, lambda: g(["H_K", "sum_d"]), "§5 dry run"),
    ("X2 bootstrap q05", -0.22, 2, lambda: g(["H_K", "X2_q05"]), "§5 dry run, §4.1 Q1"),
    ("X1 p (seed 0)", 0.076, 3, lambda: g(["H_K", "X1"])[0]["p"], "§4.1 Q1"),
    ("p_order", 0.040, 3, lambda: g(["p_order", "p"]), "§5"),
    ("X̄_K", 2.6, 1, lambda: g(["p_order", "Xbar_K"]), "§5"),
    ("X̄_oracle mean", 0.92, 2, lambda: g(["p_order", "Xbar_oracle_mean"]), "§5"),
    ("H-K1 sum d", 13.3, 1, lambda: g(["H_K1", "sum_d"]), "§5 rev 6"),
    ("H-K1 bootstrap q05", 2.4, 1, lambda: g(["H_K1", "q05"]), "§5 rev 6"),
    ("H-R mean of persons", 1.02, 2, lambda: g(["H_R", "R", "mean"]), "§4.3"),
    ("H-R t p", 0.17, 2, lambda: g(["H_R", "R", "t_p"]), "§4.3"),
    ("H-R 90 % CI low", 0.63, 2, lambda: g(["H_R", "R", "ci90"])[0], "§4.3"),
    ("H-R 90 % CI high", 1.42, 2, lambda: g(["H_R", "R", "ci90"])[1], "§4.3"),
    ("R-thin mean of persons", 0.88, 2, lambda: g(["H_R", "Rthin", "mean"]), "§5 rev 3"),
    ("R-thin t p", 0.44, 2, lambda: g(["H_R", "Rthin", "t_p"]), "§5 rev 3"),
    ("R − R-thin paired", -0.01, 2, lambda: g(["H_R", "paired", "diff_mean"]), "§5 B1"),
    ("R − R-thin CI low", -0.33, 2, lambda: g(["H_R", "paired", "ci90"])[0], "§5 B1"),
    ("R − R-thin CI high", 0.31, 2, lambda: g(["H_R", "paired", "ci90"])[1], "§5 B1"),
    ("acf detrended lag 1", 0.17, 2, lambda: g(["H_R", "acf", "detrended", "pooled", "lag1", "r"]), "§5 B2"),
    ("dAlt coefficient", 0.60, 2, lambda: g(["covariates", "primary_dalt", "coef"]), "§5 S2"),
    ("dAlt CI low", 0.29, 2, lambda: g(["covariates", "primary_dalt", "ci90"])[0], "§5 S2"),
    ("dAlt CI high", 1.26, 2, lambda: g(["covariates", "primary_dalt", "ci90"])[1], "§5 S2"),
    ("dAlt with dTime", 0.76, 2, lambda: g(["covariates", "sens2_dalt_dtime", "dalt", "coef"]), "§5 rev 5"),
    ("corr(dAlt, dTime)", -0.33, 2, lambda: g(["covariates", "corr_dalt_dtime"]), "§5 rev 5"),
    ("ACI bound vacuous cells", 7, 0, lambda: g(["ACI_bound", "n_vacuous"]), "§3 D4"),
] + [(f"sensitivity {m} +{d}", v, 2, (lambda m=m, d=d: S[f"{d}|all"][f"sens_{m}"]), "§5")
     for m, vals in [("K", (0.40, 0.55, 0.55)), ("R", (0.29, 0.45, 0.50)), ("Rthin", (0.21, 0.33, 0.36)), ("Rsel", (0.48, 0.58, 0.57)),
                     ("ACI", (0.32, 0.49, 0.53)), ("K_self", (0.19, 0.21, 0.24)), ("B0m", (0.28, 0.43, 0.56))]
     for d, v in zip((20, 30, 40), vals)] + [
    (f"K − B0m mean of persons +{d}", v, 2, (lambda d=d: g(["sensitivity", "matched_diff", f"{d}|K-B0m", "mean"])), "§5 matched FAR")
    for d, v in zip((20, 30, 40), (0.11, 0.12, -0.01))]

EXPLAIN = {
    "dAlt CI high": "Monte Carlo: the new script uses 2,000 pairs-bootstrap draws (5 x 400), as prereg §4.2 d prescribes; the "
                    "dev script used 1,500 (5 x 300). With 3 clusters the upper percentile moves; the point estimate is identical.",
}
rows, n_bad = [], 0
for name, ref, dec, get, where in CHECKS:
    val = get(); tol = 0.5 * 10 ** (-dec) + 1e-12
    ok = abs(val - ref) <= tol
    n_bad += not ok
    rows.append(dict(item=name, prereg=ref, script=round(val, 4), match=ok, where=where, note=EXPLAIN.get(name, "")))

# (2) arrays vs dev cache, (3) run 1 vs run 2
KEYS = ["st", "sq", "hp3"] + [f"{p}{d}" for d in (20, 30, 40) for p in ("on", "sq")] + [f"hp3_{d}" for d in (20, 30, 40)] + ["on_cal", "sq_cal", "hp3_cal"]
arr = []
for sid in ids:
    for k in range(5):
        a = np.load(ROOT / f"02_data/derived/confirm_cache/{L1}/{sid}_seed{k}.npz"); b = np.load(ROOT / f"02_data/derived/s2_cache/WildPPG_{sid}_seed{k}.npz")
        c = np.load(ROOT / f"02_data/derived/confirm_cache/{L2}/{sid}_seed{k}.npz")
        for key in KEYS:
            arr.append(dict(sid=sid, seed=k, key=key, same_as_dev_cache=bool(np.array_equal(a[key], b[key])),
                            same_run1_run2=bool(np.array_equal(a[key], c[key]))))
j1 = (ROOT / f"05_experiments/confirm/runs/{L1}/results.json").read_bytes(); j2 = (ROOT / f"05_experiments/confirm/runs/{L2}/results.json").read_bytes()
c1 = (ROOT / f"05_experiments/confirm/runs/{L1}/conclusions.md").read_bytes(); c2 = (ROOT / f"05_experiments/confirm/runs/{L2}/conclusions.md").read_bytes()
summary = dict(n_checks=len(rows), n_mismatch=n_bad, arrays_checked=len(arr),
               arrays_equal_dev_cache=sum(x["same_as_dev_cache"] for x in arr), arrays_equal_run1_run2=sum(x["same_run1_run2"] for x in arr),
               results_json_identical=j1 == j2, conclusions_identical=c1 == c2,
               results_sha256=[hashlib.sha256(j1).hexdigest(), hashlib.sha256(j2).hexdigest()])
out = ROOT / "05_experiments/confirm/regression_check.json"
out.write_text(json.dumps(dict(summary=summary, checks=rows, arrays=[x for x in arr if not (x["same_as_dev_cache"] and x["same_run1_run2"])]), indent=1))
md = ["| Item | Prereg | Script | Match | Where | Note |", "|---|---|---|---|---|---|"]
md += [f"| {r['item']} | {r['prereg']} | {r['script']} | {'yes' if r['match'] else '**NO**'} | {r['where']} | {r['note']} |" for r in rows]
(ROOT / "05_experiments/confirm/regression_check.md").write_text("\n".join(md) + "\n\n" + json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
for r in rows:
    if not r["match"]: print("MISMATCH", r)
