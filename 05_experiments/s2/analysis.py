"""S2 round-1 analyses (Cowork review), DEV ONLY, from 02_data/derived/s2_cache (precompute.py).
    uv run python 05_experiments/s2/analysis.py
M1  sensitivity / delay at matched FAR: main vs B0 (120), B0-matched, B0-state-matched, B6 (WildPPG dev).
M2  drift of block maxima per (ID, state): Spearman(rank, time) + permutation p; figures.
m3  block fragmentation (real-time span, segments, valid D) vs block max.
m5  fraction of monitored time with an active within-person threshold per (ID, state).
m7  HMM sigma 3/5/8: MAE, recovered injection offset, tracking delay, main-method sensitivity.
C   calibration contamination (Delta = 30 in calibration block j = 2).
All methods share 04_code/h32/method.py (same windows, same debounce, same refractory, same test hours).
"""
import sys, json, pathlib, numpy as np
from scipy.stats import spearmanr
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import method as M
C = ROOT / "02_data/derived/s2_cache"; OUT = pathlib.Path(__file__).parent / "results"; FIG = pathlib.Path(__file__).parent / "figs"
TH = 0.02896360121667385; N_CAL, KSTAR = 6, 6; DELTAS = (20, 30, 40); SEEDS = range(5); WIN_DET = 270.0
STN = {0: "rest", 1: "light", 2: "vigorous"}
RECS = [("WildPPG", s) for s in ("e61", "u7y", "ngh")] + [("DaLiA", s) for s in ("S10", "S11", "S12")]


def load(src, sid, k):
    return dict(np.load(C / f"{src}_{sid}_seed{k}.npz"))


def stream(st, sq, hp):
    mon = sq >= TH; blk = M.blocks(mon, st); D = M.debounce(hp, mon, st, blk)
    S = M.block_scores(D, st, blk); nc = M.complete_blocks(mon, st, blk)
    return dict(mon=mon, blk=blk, D=D, S=S, nc=nc)


def eligible(t0, st, tau, cal_end):
    """injected event counts iff its onset state has a within-person threshold and calibration is over."""
    kk = int(st[min(int((t0 - 4) // 2), len(st) - 1)])
    return tau[kk] is not None and t0 > cal_end[kk]


def cal_end_time(st, blk, k):
    idx = np.flatnonzero((st == k) & (blk == N_CAL - 1))
    return 2.0 * idx[-1] + 4 if len(idx) else np.inf


def test_block_max(z, st, tm):
    """per test block: (state, block, max D over its test windows)."""
    sel = np.flatnonzero(tm)
    out = {}
    for i in sel:
        key = (int(st[i]), int(z["blk"][i])); out[key] = max(out.get(key, -np.inf), z["D"][i])
    return out


def detect(alarm_idx, onsets):
    c = 2.0 * alarm_idx + 4
    res = []
    for t0 in onsets:
        hit = c[(c >= t0) & (c <= t0 + WIN_DET)]
        res.append((len(hit) > 0, float(hit[0] - t0) if len(hit) else np.nan))
    return res


# ---------------------------------------------------------------- per (rec, seed), sigma = 3
REC = {}
for src, sid in RECS:
    for k in SEEDS:
        z = load(src, sid, k); st = z["st"]
        cl = stream(st, z["sq"], z["hp3"])
        tau = M.main_thresholds(cl["S"], cl["nc"], N_CAL, KSTAR)
        tm = M.test_mask(st, cl["blk"], tau, cl["nc"], N_CAL)
        REC[(src, sid, k)] = dict(z=z, st=st, cl=cl, tau=tau, tm=tm, H=M.monitored_hours(cl["mon"], tm),
                                  tbm=test_block_max(cl, st, tm),
                                  cal_end={kk: cal_end_time(st, cl["blk"], kk) for kk in range(3)})
print("loaded", len(REC), flush=True)

# ---------------------------------------------------------------- m5 coverage of the within-person threshold
m5 = []
for (src, sid, k), r in REC.items():
    if k: continue
    st, cl = r["st"], r["cl"]; n = len(st)
    for kk in range(3):
        mon_k = cl["mon"] & (st == kk)
        m5.append(dict(src=src, sid=sid, state=STN[kk], rec_h=n * 2 / 3600, monitored_h=float(mon_k.sum() * 2 / 3600),
                       complete_blocks=cl["nc"][kk], calibrated=r["tau"][kk] is not None,
                       cal_done_at_h=float(r["cal_end"][kk] / 3600) if np.isfinite(r["cal_end"][kk]) else None,
                       active_h=float((mon_k & r["tm"]).sum() * 2 / 3600),
                       frac_active_of_state=float((mon_k & r["tm"]).sum() / max(mon_k.sum(), 1)),
                       frac_active_of_wear=float((mon_k & r["tm"]).sum() / n)))
print("m5 done", flush=True)

# ---------------------------------------------------------------- M1 matched FAR (WildPPG; DaLiA has no test blocks)
def thr_arrays(r, method, par):
    st = r["st"]
    if method == "main": return M.thr_from_state(st, r["tau"])
    if method == "B0": return np.full(len(st), 120.0)
    if method == "B0m": return np.full(len(st), par)
    if method == "B0sm": return M.thr_from_state(st, par)
    if method == "B6":
        cl = r["cl"]; t = {}
        for kk in range(3):
            if r["tau"][kk] is None: t[kk] = None; continue
            v = cl["D"][(st == kk) & (cl["blk"] >= 0) & (cl["blk"] < N_CAL) & np.isfinite(cl["D"])]
            t[kk] = float(np.quantile(v, par))
        return M.thr_from_state(st, t)


def count_alarms(r, thr):
    return sum(1 for (kk, b), mx in r["tbm"].items() if mx > (thr if np.isscalar(thr) else thr[kk]))


WR = lambda k: [REC[("WildPPG", s, k)] for s in ("e61", "u7y", "ngh")]
match = {}
for k in SEEDS:
    rs = WR(k); H = sum(r["H"] for r in rs)
    tgt = sum(count_alarms(r, {kk: (r["tau"][kk] if r["tau"][kk] is not None else np.inf) for kk in range(3)}) for r in rs)
    tgt_k = {kk: sum(sum(1 for (s_, b), mx in r["tbm"].items() if s_ == kk and r["tau"][kk] is not None and mx > r["tau"][kk]) for r in rs) for kk in range(3)}
    allmx = np.sort(np.concatenate([[mx for (s_, b), mx in r["tbm"].items()] for r in rs]))[::-1]
    c = float((allmx[tgt - 1] + allmx[tgt]) / 2) if 0 < tgt < len(allmx) else (float(allmx[0]) + 1 if tgt == 0 else -np.inf)
    ck = {}
    for kk in range(3):
        v = np.sort([mx for r in rs for (s_, b), mx in r["tbm"].items() if s_ == kk])[::-1]; t = tgt_k[kk]
        ck[kk] = (float((v[t - 1] + v[t]) / 2) if 0 < t < len(v) else (float(v[0]) + 1 if (t == 0 and len(v)) else None)) if len(v) else None
    best = None
    calv = [{kk: r["cl"]["D"][(r["st"] == kk) & (r["cl"]["blk"] >= 0) & (r["cl"]["blk"] < N_CAL) & np.isfinite(r["cl"]["D"])]
             for kk in range(3) if r["tau"][kk] is not None} for r in rs]
    for q in np.arange(0.50, 1.0001, 0.0025):
        cnt = 0
        for r, cv in zip(rs, calv):
            t = {kk: float(np.quantile(v, q)) for kk, v in cv.items()}
            cnt += sum(1 for (kk, b), mx in r["tbm"].items() if mx > t[kk])
        if best is None or abs(cnt - tgt) < abs(best[1] - tgt): best = (float(q), cnt)
    match[k] = dict(H=H, main_count=tgt, main_far=tgt / H, B0m_c=c, B0sm_c=ck, B6_q=best[0], B6_count=best[1],
                    B0_count=sum(count_alarms(r, 120.0) for r in rs), main_count_by_state=tgt_k)
    print("seed", k, "H %.1f main FAR %.2f/h (%d) | B0 %.2f/h | B0m c=%.1f | B0sm %s | B6 q=%.4f (%d)" % (
        H, tgt / H, tgt, match[k]["B0_count"] / H, c, {STN[a]: (None if b is None else round(b, 1)) for a, b in ck.items()}, best[0], best[1]), flush=True)

METHODS = ["main", "B0", "B0m", "B0sm", "B6"]
ev = []
for k in SEEDS:
    for r, sid in zip(WR(k), ("e61", "u7y", "ngh")):
        z, st = r["z"], r["st"]
        par = dict(main=None, B0=None, B0m=match[k]["B0m_c"], B0sm=match[k]["B0sm_c"], B6=match[k]["B6_q"])
        for d in DELTAS:
            inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"])
            tm = M.test_mask(st, inj["blk"], r["tau"], inj["nc"], N_CAL)
            ons = [t for t in z[f"on{d}"] if eligible(t, st, r["tau"], r["cal_end"])]
            det = {}
            for mth in METHODS:
                thr = thr_arrays(r, mth, par[mth])
                det[mth] = detect(M.alarms(inj["D"], st, inj["blk"], thr, tm), ons)
            for j, t0 in enumerate(ons):
                i0 = min(int((t0 - 4) // 2), len(st) - 1)
                ev.append(dict(seed=k, sid=sid, delta=d, t0=float(t0), state=STN[int(st[i0])],
                               gated_at_onset=bool(z[f"sq{d}"][i0] < TH),
                               **{f"det_{m_}": bool(det[m_][j][0]) for m_ in METHODS},
                               **{f"delay_{m_}": det[m_][j][1] for m_ in METHODS}))
print("events", len(ev), flush=True)


def boot_diff(rows, a, b, B=2000):
    """event bootstrap stratified by (sid, seed); paired difference of detection rates a - b."""
    rng = np.random.default_rng(0); strata = {}
    for x in rows: strata.setdefault((x["sid"], x["seed"]), []).append(x)
    arr = [(np.array([x[f"det_{a}"] for x in v], float), np.array([x[f"det_{b}"] for x in v], float)) for v in strata.values()]
    out = []
    for _ in range(B):
        sa = sb = nn = 0
        for xa, xb in arr:
            ii = rng.integers(0, len(xa), len(xa)); sa += xa[ii].sum(); sb += xb[ii].sum(); nn += len(xa)
        out.append((sa - sb) / nn)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


m1 = {}
for d in DELTAS:
    for stt in ("all", "rest", "light", "vigorous"):
        rows = [x for x in ev if x["delta"] == d and (stt == "all" or x["state"] == stt)]
        if not rows: continue
        e = dict(n_events=len(rows), frac_gated_at_onset=float(np.mean([x["gated_at_onset"] for x in rows])))
        for m_ in METHODS:
            e[f"sens_{m_}"] = float(np.mean([x[f"det_{m_}"] for x in rows]))
            dl = [x[f"delay_{m_}"] for x in rows if x[f"det_{m_}"]]
            e[f"delay_med_{m_}"] = float(np.median(dl)) if dl else None
        e["diff_main_minus_B0m"] = e["sens_main"] - e["sens_B0m"]; e["diff_ci"] = boot_diff(rows, "main", "B0m")
        e["diff_main_minus_B0sm"] = e["sens_main"] - e["sens_B0sm"]; e["diff_ci_B0sm"] = boot_diff(rows, "main", "B0sm")
        m1[f"{d}|{stt}"] = e
        print("M1 +%d %-8s n=%4d | sens main %.2f B0 %.2f B0m %.2f B0sm %.2f B6 %.2f | main-B0m %+.3f [%+.3f, %+.3f] | main-B0sm %+.3f [%+.3f, %+.3f] | delay main %s B0m %s" % (
            d, stt, len(rows), e["sens_main"], e["sens_B0"], e["sens_B0m"], e["sens_B0sm"], e["sens_B6"],
            e["diff_main_minus_B0m"], *e["diff_ci"], e["diff_main_minus_B0sm"], *e["diff_ci_B0sm"], e["delay_med_main"], e["delay_med_B0m"]), flush=True)

# ---------------------------------------------------------------- M2 drift
rng = np.random.default_rng(0); m2 = []
for (src, sid, k), r in REC.items():
    for kk in range(3):
        S = r["cl"]["S"][kk][:r["cl"]["nc"][kk]]; S = S[np.isfinite(S)]
        if len(S) < 6: m2.append(dict(src=src, sid=sid, seed=k, state=STN[kk], n_blocks=len(S), rho=None, p=None)); continue
        rho = spearmanr(np.arange(len(S)), S).statistic
        perm = np.array([spearmanr(np.arange(len(S)), rng.permutation(S)).statistic for _ in range(2000)])
        p = float((1 + np.sum(np.abs(perm) >= abs(rho))) / (1 + len(perm)))
        m2.append(dict(src=src, sid=sid, seed=k, state=STN[kk], n_blocks=len(S), rho=float(rho), p=p))
print("M2 done", flush=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for src in ("WildPPG", "DaLiA"):
    sids = [s for (a, s) in RECS if a == src]
    fig, ax = plt.subplots(len(sids), 3, figsize=(12, 2.6 * len(sids)), squeeze=False)
    for i, sid in enumerate(sids):
        r = REC[(src, sid, 0)]; st, blk = r["st"], r["cl"]["blk"]
        for kk in range(3):
            S = r["cl"]["S"][kk][:r["cl"]["nc"][kk]]
            t = [2.0 * np.flatnonzero((st == kk) & (blk == j))[0] / 3600 for j in range(len(S))]
            a = ax[i, kk]; a.plot(t, S, "o-", ms=3, lw=0.8, color="#3b6ea5")
            if len(S) >= N_CAL: a.plot(t[:N_CAL], S[:N_CAL], "o", ms=4, color="#c0504d", label="calibration")
            if r["tau"][kk] is not None: a.axhline(r["tau"][kk], ls="--", lw=0.8, color="#c0504d")
            row = [x for x in m2 if x["src"] == src and x["sid"] == sid and x["seed"] == 0 and x["state"] == STN[kk]][0]
            a.set_title(f"{sid} · {STN[kk]} · n={row['n_blocks']}" + ("" if row["rho"] is None else f" · ρ={row['rho']:.2f}, p={row['p']:.3f}"), fontsize=8)
            a.set_xlabel("time (h)", fontsize=7); a.set_ylabel("block max (bpm)", fontsize=7); a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"M2_blockmax_{src}.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- m3 fragmentation
m3 = []
for (src, sid, k), r in REC.items():
    if k: continue
    st, cl = r["st"], r["cl"]
    for kk in range(3):
        for j in range(cl["nc"][kk]):
            idx = np.flatnonzero((st == kk) & (cl["blk"] == j))
            seg = 1 + int(np.sum(np.diff(idx) > 1)); span = (idx[-1] - idx[0] + 1) * 2 / 60
            m3.append(dict(src=src, sid=sid, state=STN[kk], block=j, span_min=span, segments=seg,
                           valid_D=int(np.isfinite(cl["D"][idx]).sum()), smax=float(cl["S"][kk][j])))
m3sum = {}
for src in ("WildPPG", "DaLiA"):
    for stt in STN.values():
        x = [b for b in m3 if b["src"] == src and b["state"] == stt and np.isfinite(b["smax"])]
        if len(x) < 5: continue
        sp = np.array([b["span_min"] for b in x]); sg = np.array([b["segments"] for b in x]); vd = np.array([b["valid_D"] for b in x]); sm = np.array([b["smax"] for b in x])
        m3sum[f"{src}|{stt}"] = dict(n=len(x), span_min_q=[float(v) for v in np.percentile(sp, [10, 50, 90])],
                                     segments_q=[float(v) for v in np.percentile(sg, [10, 50, 90])],
                                     validD_q=[float(v) for v in np.percentile(vd, [10, 50, 90])],
                                     spearman_segments_smax=float(spearmanr(sg, sm).statistic),
                                     spearman_validD_smax=float(spearmanr(vd, sm).statistic),
                                     spearman_span_smax=float(spearmanr(sp, sm).statistic))
print("m3 done", flush=True)

# ---------------------------------------------------------------- m7 sigma sensitivity
m7 = {}
for sg in (3, 5, 8):
    rows = []
    for (src, sid, k), r in REC.items():
        z, st = r["z"], r["st"]; hp = z[f"hp{sg}"]; ref = z["ref"]
        if src == "WildPPG":
            j = np.arange(0, len(hp), 4); ok = np.isfinite(ref[j]); mae = float(np.mean(np.abs(hp[j][ok] - ref[j][ok])))
        else:
            n_ = min(len(hp), len(ref)); mae = float(np.mean(np.abs(hp[:n_] - ref[:n_])))
        cl = stream(st, z["sq"], hp); tau = M.main_thresholds(cl["S"], cl["nc"], N_CAL, KSTAR)
        rec = {}
        for d in DELTAS:
            c = np.arange(len(hp)) * 2 + 4.0; h = z[f"hp{sg}_{d}"]; rv, dl = [], []
            for t0 in z[f"on{d}"]:
                pl = (c > t0 + 30) & (c < t0 + 210); pre = (c > t0 - 60) & (c < t0)
                if pl.sum() < 10 or pre.sum() < 5: continue
                base = np.median(h[pre]); rv.append(float(np.median(h[pl]) - base))
                post = np.flatnonzero((c >= t0) & (c <= t0 + 210) & (h - base >= d / 2))
                dl.append(float(c[post[0]] - t0) if len(post) else np.nan)
            inj = stream(st, z[f"sq{d}"], h); tm = M.test_mask(st, inj["blk"], tau, inj["nc"], N_CAL)
            ce = {kk: cal_end_time(st, cl["blk"], kk) for kk in range(3)}
            ons = [t for t in z[f"on{d}"] if eligible(t, st, tau, ce)]
            dt = detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau), tm), ons)
            rec[d] = dict(rec=rv, delay=dl, det=[x[0] for x in dt], det_delay=[x[1] for x in dt if x[0]])
        clm = M.test_mask(st, cl["blk"], tau, cl["nc"], N_CAL)
        far_n = len(M.alarms(cl["D"], st, cl["blk"], M.thr_from_state(st, tau), clm)); far_h = M.monitored_hours(cl["mon"], clm)
        rows.append(dict(src=src, sid=sid, seed=k, mae=mae, rec=rec, far_n=far_n, far_h=far_h))
    out = {}
    for src in ("WildPPG", "DaLiA"):
        x = [r for r in rows if r["src"] == src]
        o = dict(mae={sid: float(np.mean([r["mae"] for r in x if r["sid"] == sid])) for sid in sorted({r["sid"] for r in x})})
        for d in DELTAS:
            rv = np.concatenate([r["rec"][d]["rec"] for r in x]); dl = np.concatenate([r["rec"][d]["delay"] for r in x])
            det = np.concatenate([r["rec"][d]["det"] for r in x]) if any(len(r["rec"][d]["det"]) for r in x) else np.array([])
            o[f"inj{d}"] = dict(recovered_median=float(np.median(rv)), track_delay_med_s=float(np.nanmedian(dl)), frac_never_half=float(np.mean(np.isnan(dl))),
                                sens_main=float(det.mean()) if len(det) else None, n_events_test=int(len(det)))
        H = sum(r["far_h"] for r in x); o["main_far"] = (sum(r["far_n"] for r in x) / H) if H > 0 else None; o["main_test_h"] = H
        out[src] = o
    m7[sg] = out
    print("m7 sigma", sg, json.dumps(out)[:600], flush=True)

# ---------------------------------------------------------------- contamination of calibration
cont = []
for (src, sid, k), r in REC.items():
    z, st = r["z"], r["st"]
    if len(z["on_cal"]) == 0: continue
    cc = stream(st, z["sq_cal"], z["hp3_cal"]); tau_c = M.main_thresholds(cc["S"], cc["nc"], N_CAL, KSTAR)
    for t0 in z["on_cal"]:
        kk = int(st[int(t0 // 2)])   # on_cal = 2 i0, i0 = first window of block 2 of the contaminated state
        cont.append(dict(src=src, sid=sid, seed=k, state=STN[kk], tau_clean=r["tau"][kk], tau_cont=tau_c[kk],
                         d_tau=(None if r["tau"][kk] is None or tau_c[kk] is None else tau_c[kk] - r["tau"][kk])))
    if src == "WildPPG":
        for d in DELTAS:
            inj = stream(st, z[f"sq{d}"], z[f"hp3_{d}"]); tm = M.test_mask(st, inj["blk"], r["tau"], inj["nc"], N_CAL)
            ons = [t for t in z[f"on{d}"] if eligible(t, st, r["tau"], r["cal_end"])]
            a0 = detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, r["tau"]), tm), ons)
            tau_mix = {kk: (tau_c[kk] if tau_c[kk] is not None else r["tau"][kk]) for kk in range(3)}
            a1 = detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau_mix), tm), ons)
            cont.append(dict(src=src, sid=sid, seed=k, delta=d, sens_clean_cal=float(np.mean([x[0] for x in a0])) if a0 else None,
                             sens_cont_cal=float(np.mean([x[0] for x in a1])) if a1 else None, n=len(ons)))
print("contamination done", flush=True)

# ---------------------------------------------------------------- m7b delay vs onset speed (ramp 10/30/60 s, Delta 30)
m7b = {}
for sg in (3, 5, 8):
    for src in ("WildPPG", "DaLiA"):
        for ramp in (10, 30, 60):
            tdl, ddl, det = [], [], []
            for (s_, sid, k), r in REC.items():
                if s_ != src: continue
                z, st = r["z"], r["st"]
                if ramp == 30:
                    on, h, sq = z["on30"], z[f"hp{sg}_30"], z["sq30"]
                else:
                    f = C / f"ramp_{src}_{sid}_seed{k}.npz"
                    if not f.exists(): continue
                    zr = np.load(f); on, h, sq = zr[f"on_r{ramp}"], zr[f"hp{sg}_r{ramp}"], zr[f"sq_r{ramp}"]
                c = np.arange(len(h)) * 2 + 4.0
                for t0 in on:
                    pre = (c > t0 - 60) & (c < t0)
                    if pre.sum() < 5: continue
                    base = np.median(h[pre]); post = np.flatnonzero((c >= t0) & (c <= t0 + 2 * ramp + 180) & (h - base >= 15))
                    tdl.append(float(c[post[0]] - t0) if len(post) else np.nan)
                cl = stream(st, z["sq"], z[f"hp{sg}"]); tau = M.main_thresholds(cl["S"], cl["nc"], N_CAL, KSTAR)
                ce = {kk: cal_end_time(st, cl["blk"], kk) for kk in range(3)}
                inj = stream(st, sq, h); tm = M.test_mask(st, inj["blk"], tau, inj["nc"], N_CAL)
                ons = [t for t in on if eligible(t, st, tau, ce)]
                for hit, dl in detect(M.alarms(inj["D"], st, inj["blk"], M.thr_from_state(st, tau), tm), ons):
                    det.append(hit); ddl.append(dl)
            if not tdl: continue
            m7b[f"sigma{sg}|{src}|ramp{ramp}"] = dict(n_events=len(tdl), track_delay_to_half_med_s=float(np.nanmedian(tdl)),
                                                    frac_never_half=float(np.mean(np.isnan(tdl))), n_test_events=len(det),
                                                    sens_main=float(np.mean(det)) if det else None,
                                                    det_delay_med_s=float(np.nanmedian(ddl)) if det and any(det) else None)
            print("m7b", f"sigma{sg}|{src}|ramp{ramp}", m7b[f"sigma{sg}|{src}|ramp{ramp}"], flush=True)

json.dump(dict(m7b=m7b, match={str(k): v for k, v in match.items()}, m1=m1, m2=m2, m3=m3sum, m5=m5, m7={str(k): v for k, v in m7.items()},
               contamination=cont, events=ev), open(OUT / "s2_analysis.json", "w"), indent=1, default=float)
print("written")
