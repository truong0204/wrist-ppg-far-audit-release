"""Monte-Carlo check of Proposition 2 (03_theory/R1.md) on synthetic m-dependent MA(m) block scores.
S_j = sum_{l=0}^{m} Z_{j-l}, Z iid N(0,1): stationary, m-dependent, continuous. Seeds 0-4, 20,000 runs each.
Checks: (2a) P(rate > (60/L) p*) <= delta_1 (exact equality expected); (2b) P(N(T) > B) <= delta_1 + delta_2.
Also shows what happens WITHOUT thinning (all n blocks used, iid formula) -> anticonservative when m > 0.
    uv run python 05_experiments/t4/R1_sim.py
"""
import json, math, pathlib, numpy as np
from scipy.stats import beta, binom, norm
OUT = pathlib.Path(__file__).parent / "results"; L = 10; bph = 60 / L; a = 2 * L / 60; d1 = d2 = 0.05; R = 20000
res = []
for m, n, T in [(0, 12, 60), (1, 12, 60), (1, 24, 60), (2, 24, 60)]:
    n1 = math.ceil(n / (m + 1))
    k_rate = min([k for k in range(1, n1 + 1) if beta.cdf(a, n1 + 1 - k, k) >= 0.90], default=n1)
    p_star = beta.ppf(1 - d1, n1 + 1 - k_rate, k_rate)
    Tc = [len(range(c, T, m + 1)) for c in range(m + 1)]
    B = sum(binom.ppf(1 - d2 / (m + 1), t, p_star) for t in Tc)
    sd = math.sqrt(m + 1)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((R, n + m + T + m))
        S = sum(Z[:, m - l: m - l + n + m + T] for l in range(m + 1))          # shape R x (n + g + T), g = m
        cal, test = S[:, :n], S[:, n + m:]
        thin = np.sort(cal[:, ::m + 1], 1); tau = thin[:, k_rate - 1]
        rate = bph * norm.sf(tau / sd)
        N = (test > tau[:, None]).sum(1)
        k_all = min([k for k in range(1, n + 1) if beta.cdf(a, n + 1 - k, k) >= 0.90], default=n)
        tau_all = np.sort(cal, 1)[:, k_all - 1]
        res.append(dict(m=m, n=n, T=T, seed=seed, k=k_rate, n_thin=n1,
                        p_rate_gt_pstar=float(np.mean(rate > bph * p_star)), delta1=d1,
                        p_rate_gt_r=float(np.mean(rate > 2.0)), target_rate_conf=0.90,
                        p_count_gt_B=float(np.mean(N > B)), delta12=d1 + d2, B=float(B),
                        no_thin_p_rate_gt_r=float(np.mean(bph * norm.sf(tau_all / sd) > 2.0)), mean_rate=float(rate.mean())))
    x = [r for r in res if (r["m"], r["n"]) == (m, n)]
    f = lambda k: (np.mean([r[k] for r in x]), np.std([r[k] for r in x]))
    print("m=%d n=%d T=%d k*=%d/%d | P(rate>r)=%.3f±%.3f (<=0.10) | P(rate>p*)=%.3f (=0.05) | P(N>B=%d)=%.3f (<=0.10) | no thinning P(rate>r)=%.3f | E rate %.2f/h" % (
        m, n, T, k_rate, n1, *f("p_rate_gt_r"), f("p_rate_gt_pstar")[0], B, f("p_count_gt_B")[0], f("no_thin_p_rate_gt_r")[0], f("mean_rate")[0]))
json.dump(res, open(OUT / "R1_sim.json", "w"), indent=1)
