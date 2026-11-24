"""S2 r2 (PI decision 22-11, items 2-4). DEV ONLY: WildPPG e61/u7y/ngh, sigma 3, E2 seeds 0-4.
Seeds are NOT independent replicates: every number is computed per seed, then AVERAGED over seeds; n = persons/cells.
Block-level counting uses Lemma 1 (alarms in complete test blocks = #{test blocks with S_j > tau}); checked against
the window-level alarm code on every cell (assert).

2a K-oracle (control, NOT deployable): calibration = 6 blocks drawn at random from all complete blocks of the cell,
   test = the other blocks; 2,000 draws per cell and seed. For distinct block maxima the oracle split is exchangeable
   by construction, so N ~ BetaBinomial(T, 1, 6) exactly: the oracle "keeps the guarantee" by construction. Its use is
   as the reference distribution given the observed block values: permutation p = P(X_oracle >= X_K), where
   X = #cells with rate > r and cells are drawn independently. Draw b uses the SAME random keys in every E2 seed
   (common random numbers), so the seed-averaged oracle keeps its between-draw variance.
2b K-intact is infeasible on dev (0 of 203 complete blocks is one real-time segment; no real-time run of one state
   reaches 10 min). SUBSTITUTE (needs PI approval): per cell, Spearman of fragmentation (valid-D count, number of
   real-time segments) vs block index, and partial Spearman of block max vs block index given both fragmentation
   covariates (rank residuals).
2c drift index rho_c = Spearman(S_j, j) over all complete blocks; cell-level association: Spearman(rho_c, rate_c)
   over cells and mean rho in exceeding vs non-exceeding cells; permutation p over cells (9 cells, 3 persons).
3  R (rolling, n_cal 6, gap g 1) and R-sel (alarmed blocks dropped from the window, after Sim & Kim 2026):
   FAR per person (seed-averaged), person-cluster bootstrap CI; sensitivity with clean-stream thresholds and with
   self-computed thresholds on the injected stream; masking after an event by block-level simulation.
4  Per-person sensitivity main vs B0-matched (seed-averaged) from s2_analysis.json.
    uv run python 05_experiments/s2/controls.py
"""
import sys, json, pathlib, itertools, numpy as np
from scipy.stats import spearmanr, rankdata, betabinom
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"
TH = 0.02896360121667385; NC, G, R_ = 6, 1, 2.0; WIN = 270.0; STN = ["rest", "light", "vigorous"]
IDS = ("e61", "u7y", "ngh"); SEEDS = range(5); NDRAW = 2000
rng = np.random.default_rng(20261123)


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    return dict(mon=mon, blk=blk, D=D, S=M.block_scores(D, st, blk), nc=M.complete_blocks(mon, st, blk))


def tau_K(S):
    return float(np.max(S[:NC]))


def tau_R(S, nc, selective):
    """per test block j (NC+G <= j < nc): threshold and alarm flag, sequential (R-sel drops alarmed blocks)."""
    alarmed = np.zeros(nc, bool); thr = {}
    for j in range(NC + G, nc):
        cand = [i for i in range(j - G - 1, -1, -1) if not (selective and alarmed[i])][:NC]
        t = float(np.max(S[cand])); thr[j] = t; alarmed[j] = S[j] > t
    return thr, alarmed


def win_thr(st, blk, per_block):
    """window threshold array from {(k, j): tau}; inf elsewhere."""
    return np.array([per_block.get((int(st[i]), int(blk[i])), np.inf) for i in range(len(st))])


def frag(st, blk, D, k, nc):
    seg, val = [], []
    for j in range(nc):
        idx = np.flatnonzero((st == k) & (blk == j)); seg.append(1 + int(np.sum(np.diff(idx) > 1)))
        val.append(int(np.isfinite(D[idx]).sum()))
    return np.array(seg), np.array(val)


def partial_spearman(y, x, Z):
    ry, rx = rankdata(y), rankdata(x); Zr = np.column_stack([np.ones(len(y))] + [rankdata(z) for z in Z])
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]; ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


def detect(a, ons):
    c = 2.0 * a + 4
    return [bool(np.any((c >= t) & (c <= t + WIN))) for t in ons]


cells = []                     # one row per (sid, state, seed)
sens = {m: {d: [] for d in (20, 30, 40)} for m in ("K_clean", "R_clean", "Rsel_clean", "K_self", "R_self", "Rsel_self")}
evpool = {}                    # (sid, k, seed, d) -> event scores (max D over the event window, same state)
for sid, s in itertools.product(IDS, SEEDS):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
    tauK = M.main_thresholds(cl["S"], cl["nc"])
    perR, perRs = {}, {}
    for k in range(3):
        nc = cl["nc"][k]
        if tauK[k] is None: continue
        S = cl["S"][k][:nc]; T = nc - NC
        # K, block level + Lemma 1 check against window-level alarms
        NK = int(np.sum(S[NC:] > tauK[k]))
        t1 = {kk: (tauK[kk] if kk == k else None) for kk in range(3)}
        tm = M.test_mask(st, cl["blk"], t1, cl["nc"])
        assert NK == len(M.alarms(cl["D"], st, cl["blk"], M.thr_from_state(st, t1), tm)), (sid, s, k)
        # 2a oracle
        # common random numbers: draw b uses the same random keys for every E2 seed (seeds share the data, so
        # independent draws per seed would shrink the variance of the seed-averaged oracle X)
        keys = np.random.default_rng(1000 * IDS.index(sid) + k).random((NDRAW, 200))
        No = np.empty(NDRAW, int)
        for b in range(NDRAW):
            cal = np.argsort(keys[b, :nc])[:NC]; msk = np.ones(nc, bool); msk[cal] = False
            No[b] = np.sum(S[msk] > S[cal].max())
        # 2b/2c
        seg, val = frag(st, cl["blk"], cl["D"], k, nc); j = np.arange(nc)
        rho = float(spearmanr(S, j)[0]); rho_seg = float(spearmanr(seg, j)[0]); rho_val = float(spearmanr(val, j)[0])
        prho = partial_spearman(S, j, [seg, val])
        # 3 R and R-sel (block level, clean stream)
        thrR, aR = tau_R(S, nc, False); thrRs, aRs = tau_R(S, nc, True)
        for jj, t in thrR.items(): perR[(k, jj)] = t
        for jj, t in thrRs.items(): perRs[(k, jj)] = t
        TR = nc - NC - G
        cells.append(dict(sid=sid, state=STN[k], seed=s, nc=int(nc), T=int(T), N_K=NK, rate_K=6 * NK / T,
                          oracle_rate_mean=float(6 * No.mean() / T), oracle_p_exceed=float(np.mean(6 * No / T > R_)),
                          oracle_draws=(6 * No / T > R_).astype(np.int8).tolist(),
                          p0_exceed=float(betabinom.pmf(np.arange(T + 1), T, 1, 6)[6 * np.arange(T + 1) / T > R_].sum()),
                          rho=rho, rho_seg_time=rho_seg, rho_valid_time=rho_val, partial_rho=prho,
                          T_R=int(TR), N_R=int(aR.sum()), N_Rsel=int(aRs.sum()),
                          rate_R=6 * aR.sum() / TR if TR else None, rate_Rsel=6 * aRs.sum() / TR if TR else None))
    # sensitivity: clean-stream thresholds and self-computed thresholds on the injected stream
    ce = {k: (2.0 * np.flatnonzero((st == k) & (cl["blk"] == NC - 1))[-1] + 4 if tauK[k] is not None else np.inf) for k in range(3)}
    for d in (20, 30, 40):
        inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"])
        kof = lambda t: int(st[min(int((t - 4) // 2), len(st) - 1)])
        ons = [t for t in z[f"on{d}"] if tauK[kof(t)] is not None and t > ce[kof(t)]]
        tmK = M.test_mask(st, inj["blk"], tauK, inj["nc"])
        sens["K_clean"][d] += detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tauK), tmK), ons)
        for name, per in (("R_clean", perR), ("Rsel_clean", perRs)):
            th = win_thr(st, inj["blk"], per); sens[name][d] += detect(M.alarms(inj["D"], st, inj["blk"], th, np.isfinite(th)), ons)
        tauKs = M.main_thresholds(inj["S"], inj["nc"])
        tmKs = M.test_mask(st, inj["blk"], tauKs, inj["nc"])
        sens["K_self"][d] += detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tauKs), tmKs), ons)
        for name, selv in (("R_self", False), ("Rsel_self", True)):
            per = {}
            for k in range(3):
                if inj["nc"][k] >= NC + G + 1:
                    th_, _ = tau_R(inj["S"][k][:inj["nc"][k]], inj["nc"][k], selv); per.update({(k, j): t for j, t in th_.items()})
            th = win_thr(st, inj["blk"], per); sens[name][d] += detect(M.alarms(inj["D"], st, inj["blk"], th, np.isfinite(th)), ons)
        # event score pool for the masking simulation
        c = 2.0 * np.arange(len(st)) + 4
        for t in z[f"on{d}"]:
            k = kof(t); w = (c >= t) & (c <= t + WIN) & (st == k)
            e = inj["D"][w]; e = e[np.isfinite(e)]
            if len(e): evpool.setdefault((sid, k, s, d), []).append(float(e.max()))
    print(sid, "seed", s, "done", flush=True)

# ---------------- masking after an event (block-level simulation, clean S + event scores)
mask = {m: {d: {h: [0, 0, 0] for h in range(1, 9)} for d in (20, 30, 40)} for m in ("K", "R", "Rsel")}
for sid, s in itertools.product(IDS, SEEDS):
    z = np.load(C / f"WildPPG_{sid}_seed{s}.npz"); st = z["st"]; cl = stream(st, z["sq"], z["hp3"])
    for k in range(3):
        nc = cl["nc"][k]
        if nc < NC + G + 1 + 9: continue
        S = cl["S"][k][:nc]
        for d in (20, 30, 40):
            pool = np.array(evpool.get((sid, k, s, d), []))
            if len(pool) < 3: continue
            base = {"R": tau_R(S, nc, False)[0], "Rsel": tau_R(S, nc, True)[0], "K": {j: tau_K(S) for j in range(NC, nc)}}
            for j0 in range(NC + G, nc - 8):
                for _ in range(10):
                    e1, e2 = rng.choice(pool), rng.choice(pool)
                    S1 = S.copy(); S1[j0] = max(S[j0], e1)
                    alt = {"R": tau_R(S1, nc, False)[0], "Rsel": tau_R(S1, nc, True)[0], "K": base["K"]}
                    for m in ("K", "R", "Rsel"):
                        for h in range(1, 9):
                            tb, ta = base[m][j0 + h], alt[m][j0 + h]
                            rec = mask[m][d][h]; rec[0] += e2 > tb; rec[1] += e2 > ta; rec[2] += 1
masking = {m: {d: {h: dict(det_no_prior=v[0] / v[2], det_after_event=v[1] / v[2], n=v[2]) for h, v in dd.items() if v[2]}
               for d, dd in md.items()} for m, md in mask.items()}

# ---------------- aggregation: seed average, persons as units
def seedavg(key, sid=None, state=None):
    rows = [c for c in cells if (sid is None or c["sid"] == sid) and (state is None or c["state"] == state)]
    return float(np.mean([np.mean([r[key] for r in rows if r["seed"] == s]) for s in SEEDS]))


def person_rates(nk, tk):
    out = {}
    for sid in IDS:
        N = np.mean([sum(c[nk] for c in cells if c["sid"] == sid and c["seed"] == s) for s in SEEDS])
        T = np.mean([sum(c[tk] for c in cells if c["sid"] == sid and c["seed"] == s) for s in SEEDS])
        out[sid] = (float(N), float(T) / 6)                    # (mean FA, mean monitored test hours)
    return out


def cluster_boot(pr, B=10000):
    ids = list(pr); Ns = np.array([pr[i][0] for i in ids]); Hs = np.array([pr[i][1] for i in ids])
    bi = rng.integers(0, len(ids), (B, len(ids))); v = Ns[bi].sum(1) / Hs[bi].sum(1)
    return float(Ns.sum() / Hs.sum()), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


far = {}
for name, nk, tk in (("K", "N_K", "T"), ("R", "N_R", "T_R"), ("Rsel", "N_Rsel", "T_R")):
    pr = person_rates(nk, tk); est, ci = cluster_boot(pr)
    far[name] = dict(pooled=est, cluster_ci=ci, per_person={k: v[0] / v[1] for k, v in pr.items()},
                     test_h_per_person={k: v[1] for k, v in pr.items()})

cell_tab = []
for sid, stn in itertools.product(IDS, STN):
    rows = [c for c in cells if c["sid"] == sid and c["state"] == stn]
    if not rows: continue
    g = lambda key: float(np.mean([r[key] for r in rows if r[key] is not None]))
    cell_tab.append(dict(sid=sid, state=stn, nc=rows[0]["nc"], T=rows[0]["T"], rate_K=g("rate_K"),
                         seeds_exceed_K=int(sum(r["rate_K"] > R_ for r in rows)), oracle_rate=g("oracle_rate_mean"),
                         oracle_p_exceed=g("oracle_p_exceed"), p0_exceed=g("p0_exceed"), rho=g("rho"),
                         rho_seg_time=g("rho_seg_time"), rho_valid_time=g("rho_valid_time"), partial_rho=g("partial_rho"),
                         rate_R=g("rate_R"), rate_Rsel=g("rate_Rsel")))

# 2a permutation p per seed, then averaged
perm = []
for s in SEEDS:
    cs = [c for c in cells if c["seed"] == s]; XK = sum(c["rate_K"] > R_ for c in cs)
    Xo = np.sum([np.array(c["oracle_draws"]) for c in cs], axis=0)
    perm.append(dict(seed=s, X_K=int(XK), E_oracle_X=float(Xo.mean()), p_perm=float((1 + np.sum(Xo >= XK)) / (len(Xo) + 1)),
                     E0_X=float(sum(c["p0_exceed"] for c in cs))))
# 2c cell-level association, per seed then averaged; permutation over cells
assoc = []
for s in SEEDS:
    cs = [c for c in cells if c["seed"] == s]; rho = np.array([c["rho"] for c in cs]); rate = np.array([c["rate_K"] for c in cs])
    ex = rate > R_; obs = spearmanr(rho, rate)[0]
    null = np.array([spearmanr(rng.permutation(rho), rate)[0] for _ in range(5000)])
    assoc.append(dict(seed=s, spearman_rho_rate=float(obs), p_one_sided=float((np.sum(null >= obs) + 1) / 5001),
                      mean_rho_exceed=float(rho[ex].mean()) if ex.any() else None, mean_rho_not=float(rho[~ex].mean())))

ev = json.load(open(OUT / "s2_analysis.json"))["events"]
m1p = {}
for d, sid in itertools.product((20, 30, 40), IDS):
    per_seed = [[e for e in ev if e["delta"] == d and e["sid"] == sid and e["seed"] == s] for s in SEEDS]
    m1p[f"{d}|{sid}"] = dict(n_events_mean=float(np.mean([len(x) for x in per_seed])),
                             **{f"sens_{m}": float(np.mean([np.mean([e[f"det_{m}"] for e in x]) for x in per_seed if x]))
                                for m in ("main", "B0", "B0m", "B0sm", "B6")})
for d in (20, 30, 40):
    v = [m1p[f"{d}|{sid}"]["sens_main"] - m1p[f"{d}|{sid}"]["sens_B0m"] for sid in IDS]
    m1p[f"{d}|mean_of_persons"] = dict(diff_main_B0m=float(np.mean(v)), per_person=v)

out = dict(cells=[{k: v for k, v in c.items() if k != "oracle_draws"} for c in cells], cell_table=cell_tab,
           oracle_perm=perm, assoc=assoc, far=far,
           sens={m: {d: float(np.mean(v)) for d, v in dd.items()} for m, dd in sens.items()},
           sens_n={d: len(sens["K_clean"][d]) for d in (20, 30, 40)}, masking=masking, m1_per_person=m1p)
json.dump(out, open(OUT / "controls.json", "w"), indent=1)

print("\n== cell table (seed-averaged)")
for c in cell_tab:
    print("%s %-8s nc %2d T %2d | K %.2f/h (%d/5 seeds>r) oracle %.2f/h P(>r) %.2f [P0 %.2f] | rho %+.2f partial %+.2f "
          "frag~t seg %+.2f valid %+.2f | R %.2f Rsel %.2f" % (c["sid"], c["state"], c["nc"], c["T"], c["rate_K"], c["seeds_exceed_K"],
          c["oracle_rate"], c["oracle_p_exceed"], c["p0_exceed"], c["rho"], c["partial_rho"], c["rho_seg_time"], c["rho_valid_time"],
          c["rate_R"], c["rate_Rsel"]))
print("== 2a oracle permutation", perm)
print("== 2c association", assoc)
print("== FAR (seed-avg, person-cluster bootstrap)", json.dumps(far, indent=None))
print("== sensitivity", json.dumps(out["sens"]), out["sens_n"])
print("== masking d=30", json.dumps(masking["K"][30]), "\nR", json.dumps(masking["R"][30]), "\nRsel", json.dumps(masking["Rsel"][30]))
print("== M1 per person", json.dumps(m1p))

# ---------------- prereg v2.2 rev 2 statistics, exactly as written (dry run on dev; 3 persons only)
def spearman_cells(rows, key):
    return float(spearmanr([r[key] for r in rows], [r["rate_K"] for r in rows])[0])


ct = cell_tab
pre = {}
# X2: d_i = mean over seeds of sum_c (1{R_c > r} - P0_c)
d_i = {sid: float(np.mean([sum((c["rate_K"] > R_) - c["p0_exceed"] for c in cells if c["sid"] == sid and c["seed"] == s)
                           for s in SEEDS])) for sid in IDS}
dv = np.array(list(d_i.values())); bi = rng.integers(0, len(IDS), (10000, len(IDS)))
pre["X2"] = dict(d_i=d_i, sum_d=float(dv.sum()), q05=float(np.quantile(dv[bi].sum(1), 0.05)))
# p_order on the seed-averaged X
Xbar_K = float(np.mean([sum(c["rate_K"] > R_ for c in cells if c["seed"] == s) for s in SEEDS]))
Xo = np.mean([np.sum([np.array(c["oracle_draws"]) for c in cells if c["seed"] == s], axis=0) for s in SEEDS], axis=0)
pre["p_order"] = dict(Xbar_K=Xbar_K, Xbar_oracle_mean=float(Xo.mean()), p=float((1 + np.sum(Xo >= Xbar_K - 1e-9)) / (len(Xo) + 1)))   # Q3 (Cowork r3)
# association with person-cluster bootstrap CI (cells of resampled persons; seed-averaged cell values)
for key in ("rho", "partial_rho"):
    obs = spearman_cells(ct, key); bs = []
    for b in bi[:2000]:
        rows = [c for i in b for c in ct if c["sid"] == IDS[i]]
        v = spearman_cells(rows, key)
        if np.isfinite(v): bs.append(v)
    pre[f"assoc_{key}"] = dict(spearman=obs, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
# H-R: one-sided, 5 % quantile of the pooled rate
prR = person_rates("N_R", "T_R"); Ns = np.array([prR[i][0] for i in IDS]); Hs = np.array([prR[i][1] for i in IDS])
pre["H_R"] = dict(pooled=float(Ns.sum() / Hs.sum()), q05=float(np.quantile(Ns[bi].sum(1) / Hs[bi].sum(1), 0.05)), bound=6 / 7)
out["prereg_dry_run"] = pre
json.dump(out, open(OUT / "controls.json", "w"), indent=1)
print("== prereg dry run (dev, 3 persons)", json.dumps(pre))
