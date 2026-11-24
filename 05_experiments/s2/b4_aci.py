"""S1e S4: B4 = ACI (Gibbs & Candes 2021) at block level, EXPLORATORY. DEV ONLY (WildPPG e61/u7y/ngh, sigma 3, seeds 0-4).
Per state cell, test blocks j >= 6 (as K):
  pool_j = all earlier complete blocks S_0..S_{j-1} of the state (expanding);
  tau_j = ceil((j + 1)(1 - a_j))-th smallest of pool_j  (+inf if the index > j, -inf if <= 0);
  err_j = 1{S_j > tau_j};  a_{j+1} = a_j + gamma (alpha* - err_j),  a_6 = alpha*.
Fixed a priori, not tuned: gamma = 0.05; alpha* = 1/7 (primary: same expectation target as R, 0.857/h) and 1/3
(secondary: r = 2/h). ACI's guarantee is on the long-run fraction of blocks with an alarm, counting blocks with true
events too, and holds without exchangeability; so its FAR is close to the target by construction.
Reported: FAR (clean stream), sensitivity with clean-stream and self thresholds, masking after an event (block level).
    uv run python 05_experiments/s2/b4_aci.py
"""
import sys, json, pathlib, itertools, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; IDS = ("e61", "u7y", "ngh"); SEEDS = range(5); WIN = 270.0; GAMMA = 0.05; NC = 6
TARGETS = {"aci_1_7": 1 / 7, "aci_1_3": 1 / 3}


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return dict(mon=mon, blk=blk, D=D, S=M.block_scores(D, st, blk), nc=M.complete_blocks(mon, st, blk))


def aci(S, nc, astar):
    a = astar; thr = {}; err = {}
    for j in range(NC, nc):
        pool = np.sort(S[:j]); k = int(np.ceil((j + 1) * (1 - a)))
        t = np.inf if k > j else (-np.inf if k <= 0 else float(pool[k - 1]))
        thr[j] = t; err[j] = S[j] > t; a = a + GAMMA * (astar - err[j])
    return thr, err


def detect(a, ons):
    c = 2.0 * a + 4
    return [bool(np.any((c >= t) & (c <= t + WIN))) for t in ons]


res = {}
for name, astar in TARGETS.items():
    NT = {sid: [[], []] for sid in IDS}; sens = {"clean": {d: [] for d in (20, 30, 40)}, "self": {d: [] for d in (20, 30, 40)}}
    pools = {}
    for sid, s in itertools.product(IDS, SEEDS):
        z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
        tauK = M.main_thresholds(cl["S"], cl["nc"]); per = {}; n_ = t_ = 0
        for k in range(3):
            nc = cl["nc"][k]
            if nc < NC + 1: continue
            thr, err = aci(cl["S"][k][:nc], nc, astar); per.update({(k, j): t for j, t in thr.items()})
            n_ += sum(err.values()); t_ += len(err)
        NT[sid][0].append(n_); NT[sid][1].append(t_)
        ce = {k: (2.0 * np.flatnonzero((st == k) & (cl["blk"] == NC - 1))[-1] + 4 if tauK[k] is not None else np.inf) for k in range(3)}
        kof = lambda t: int(st[min(int((t - 4) // 2), len(st) - 1)])
        for d in (20, 30, 40):
            inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"])
            ons = [t for t in z[f"on{d}"] if tauK[kof(t)] is not None and t > ce[kof(t)]]
            th = np.array([per.get((int(st[i]), int(inj["blk"][i])), np.inf) for i in range(len(st))])
            sens["clean"][d] += detect(M.alarms(inj["D"], st, inj["blk"], th, np.isfinite(th) | (th == -np.inf)), ons)
            ps = {}
            for k in range(3):
                nc = inj["nc"][k]
                if nc >= NC + 1: ps.update({(k, j): t for j, t in aci(inj["S"][k][:nc], nc, astar)[0].items()})
            th = np.array([ps.get((int(st[i]), int(inj["blk"][i])), np.inf) for i in range(len(st))])
            sens["self"][d] += detect(M.alarms(inj["D"], st, inj["blk"], th, np.isfinite(th) | (th == -np.inf)), ons)
            if d == 30:
                c = 2.0 * np.arange(len(st)) + 4
                for t in z["on30"]:
                    k = kof(t); e = inj["D"][(c >= t) & (c <= t + WIN) & (st == k)]; e = e[np.isfinite(e)]
                    if len(e): pools.setdefault((sid, k, s), []).append(float(e.max()))
    per_person = {sid: float(6 * np.mean(NT[sid][0]) / np.mean(NT[sid][1])) for sid in IDS}
    # masking, block level, Delta 30
    g = np.random.default_rng(3); mk = {h: [0, 0, 0] for h in range(1, 9)}
    for sid, s in itertools.product(IDS, SEEDS):
        z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
        for k in range(3):
            nc = cl["nc"][k]; S = cl["S"][k][:nc]; pool = np.array(pools.get((sid, k, s), []))
            if nc < NC + 1 + 9 or len(pool) < 3: continue
            base = aci(S, nc, astar)[0]
            for j0 in range(NC + 1, nc - 8):
                for _ in range(10):
                    e1, e2 = g.choice(pool), g.choice(pool); S1 = S.copy(); S1[j0] = max(S[j0], e1); alt = aci(S1, nc, astar)[0]
                    for h in range(1, 9):
                        r = mk[h]; r[0] += e2 > base[j0 + h]; r[1] += e2 > alt[j0 + h]; r[2] += 1
    res[name] = dict(alpha_star=astar, gamma=GAMMA, far_per_person=per_person, far_mean_of_persons=float(np.mean(list(per_person.values()))),
                     sens_clean={d: float(np.mean(v)) for d, v in sens["clean"].items()}, sens_self={d: float(np.mean(v)) for d, v in sens["self"].items()},
                     n_events={d: len(v) for d, v in sens["clean"].items()},
                     masking_d30={h: dict(no_prior=v[0] / v[2], after=v[1] / v[2], n=v[2]) for h, v in mk.items() if v[2]})
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(OUT / "b4_aci.json", "w"), indent=1)
