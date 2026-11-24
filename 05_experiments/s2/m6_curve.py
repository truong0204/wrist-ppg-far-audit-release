"""m6 (Cowork r1): (r, confidence) curve of ONE threshold, and blocks needed for r = 0.5/1/2. No data.
Rate form (R1.md Prop. 2a): P(rate <= r') = I_{r' L/60}(n'+1-k*, k*), L = 10 min.
    uv run python 05_experiments/s2/m6_curve.py
"""
import json, math, pathlib
from scipy.stats import beta
L = 10; OUT = pathlib.Path(__file__).parent / "results"
curve = {}
for name, n1, k in [("n6_k6 (primary)", 6, 6), ("n6_k5 (conformal)", 6, 5), ("n12_k11", 12, 11), ("n12_k12", 12, 12)]:
    pts = {rp: float(beta.cdf(rp * L / 60, n1 + 1 - k, k)) for rp in (0.25, 0.5, 1, 1.5, 2, 3, 4)}
    curve[name] = pts
    print(name.ljust(18), " ".join("r'=%.2g:%.2f" % (a, b) for a, b in pts.items()))
need = []
print("\nblocks needed (min n, order k*) for P(rate <= r) >= conf")
for m in (0, 1):
    for r in (0.5, 1, 2):
        row = []
        for conf in (0.8, 0.9, 0.95):
            a = r * L / 60
            res = None
            for n in range(1, 2000):
                n1 = math.ceil(n / (m + 1))
                ks = [kk for kk in range(1, n1 + 1) if beta.cdf(a, n1 + 1 - kk, kk) >= conf]
                if ks: res = (n, min(ks), n * L / 60); break
            need.append(dict(m=m, r=r, conf=conf, n=res[0], k=res[1], hours=res[2])); row.append("conf %.2f: n=%d (%.1f h), k*=%d" % (conf, *res[:1], res[2], res[1]))
        print("m=%d r=%.1f/h | " % (m, r) + " | ".join(row))
json.dump(dict(curve=curve, need=need), open(OUT / "m6_curve.json", "w"), indent=1)
