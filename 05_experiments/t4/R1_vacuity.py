"""Vacuity of Proposition 2 (03_theory/R1.md): per-state FAR bound per monitored hour, no data.
    uv run python 05_experiments/t4/R1_vacuity.py
alpha = r L / 60 with r = 2/h, L = 10 min; delta_1 = delta_2 = 0.05 (total 0.10).
(i) default k* = ceil((n'+1)(1-alpha)): high-probability FAR bound; (ii) PAC k*: smallest k* whose bound <= r.
(iii) rate form (Prop. 2a, SPC exceedance-probability criterion): conditional in-control rate (60/L) p(tau) <= r
     w.p. >= 1 - delta, delta in {0.10, 0.05}: smallest k* achieving it, and minimal n per m.
"""
import json, math, pathlib
from scipy.stats import beta, binom
L, r, d1, d2 = 10, 2.0, 0.05, 0.05; a = r * L / 60; bph = 60 / L
OUT = pathlib.Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)


def bound(n, m, T, k):
    n1 = math.ceil(n / (m + 1))
    if not 1 <= k <= n1:
        return None
    p_star = beta.ppf(1 - d1, n1 + 1 - k, k)                      # P(p(tau) > p*) = d1
    Tc = [len(range(c, T, m + 1)) for c in range(m + 1)]
    B = sum(binom.ppf(1 - d2 / (m + 1), t, p_star) for t in Tc)
    return dict(n_thin=n1, k=k, p_star=float(p_star), B=float(B), far_bound=float(bph * B / T),
                far_expected=float(bph * (n1 + 1 - k) / (n1 + 1)))


rows = []
for m in (0, 1, 2):
    for n in (6, 12, 18, 24, 36, 48):
        for T in (30, 60):
            n1 = math.ceil(n / (m + 1)); kd = math.ceil((n1 + 1) * (1 - a))
            dflt = bound(n, m, T, kd)
            pac = next((b for k in range(1, n1 + 1) if (b := bound(n, m, T, k)) and b["far_bound"] <= r), None)
            rows.append(dict(m=m, n_cal=n, cal_min=n * L, T=T, default=dflt, pac=pac))
            print("m=%d n=%2d (%3d min) T=%d | default k*=%s FAR<=%.2f/h (E %.2f) | PAC: %s" % (
                m, n, n * L, T, kd, dflt["far_bound"] if dflt else float("nan"), dflt["far_expected"] if dflt else float("nan"),
                "none" if pac is None else "k*=%d/%d FAR<=%.2f/h, E[FAR] %.2f" % (pac["k"], pac["n_thin"], pac["far_bound"], pac["far_expected"])))
json.dump(dict(L=L, r=r, delta1=d1, delta2=d2, rows=rows), open(OUT / "R1_vacuity.json", "w"), indent=1)

print("\nProp. 2a (rate form): P((60/L) p(tau) <= r) = I_a(n'+1-k*, k*)")
rate = []
for m in (0, 1, 2):
    for conf in (0.90, 0.95):
        for n in range(1, 200):
            n1 = math.ceil(n / (m + 1))
            ks = [k for k in range(1, n1 + 1) if beta.cdf(a, n1 + 1 - k, k) >= conf]
            if ks:
                k = min(ks); rate.append(dict(m=m, conf=conf, n_min=n, cal_min=n * L, k=k, n_thin=n1,
                                              far_expected=bph * (n1 + 1 - k) / (n1 + 1)))
                print("m=%d conf %.2f: n_min=%d blocks (%d min of that state), k*=%d/%d, E[rate]=%.2f/h" % (
                    m, conf, n, n * L, k, n1, bph * (n1 + 1 - k) / (n1 + 1)))
                break
    for n in (12, 24, 36):
        n1 = math.ceil(n / (m + 1)); ks = [k for k in range(1, n1 + 1) if beta.cdf(a, n1 + 1 - k, k) >= 0.90]
        if not ks:
            print("   m=%d n=%d: none" % (m, n)); continue
        k = min(ks); rate.append(dict(m=m, conf=0.90, n=n, k=k, n_thin=n1, far_expected=bph * (n1 + 1 - k) / (n1 + 1)))
        print("   m=%d n=%d: k*=%d/%d, E[rate]=%.2f/h (target 2)" % (m, n, k, n1, bph * (n1 + 1 - k) / (n1 + 1)))
json.dump(dict(L=L, r=r, delta1=d1, delta2=d2, rows=rows, rate_form=rate), open(OUT / "R1_vacuity.json", "w"), indent=1)
