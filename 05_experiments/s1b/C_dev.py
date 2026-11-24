"""S1b task C on DEVELOPMENT subjects S10–S12 only (C1 E2 + LOSO tuning, C2 B0, C3 B2', C4 autocorrelation).
Confirmatory subjects are touched only through HR-ECG label lengths for C4 hours (label-only, allowed).

    uv run python 05_experiments/s1b/C_dev.py
Deterministic (no random step).
"""
from __future__ import annotations
import itertools, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia, e1, e2, events, stats  # noqa: E402

THR = 120; EV = dict(k=3, gap=5, tol=10)
GRIDS = dict(sigma=[4.0, 8.0, 12.0, 16.0, 24.0], w=[0.5, 1.0], gamma=[1.0, 2.0])   # v2 grid: v1 optimum sat on the sigma edge


def eval_alarm(hr_ecg, alarm):
    ref = events.episodes(hr_ecg > THR, EV["k"], EV["gap"])
    al = events.episodes(alarm, EV["k"], EV["gap"])
    fa, det = events.match(al, ref, EV["tol"])
    return dict(n_ref=len(ref), n_alarm=len(al), n_fa=fa, n_det=det)


def pooled(rows, hours):
    fa = sum(r["n_fa"] for r in rows); H = sum(hours)
    rate, lo, hi = stats.poisson_rate_ci(fa, H)
    return dict(n_fa=fa, hours=H, far=rate, ci=[lo, hi], det=sum(r["n_det"] for r in rows),
                n_ref=sum(r["n_ref"] for r in rows), per_subject_far=[r["n_fa"] / h for r, h in zip(rows, hours)])


def main():
    D = {s: io_dalia.load_subject(s) for s in io_dalia.DEV}
    n = {s: len(D[s]["label"]) for s in D}
    hours = {s: ((n[s] - 1) * 2 + 8) / 3600 for s in D}
    # ---- C1: grid over E2 params, leave-one-dev-subject-out selection
    configs = [dict(zip(GRIDS, v)) for v in itertools.product(*GRIDS.values())]
    mae = {}
    for ci, c in enumerate(configs):
        for s in D:
            h = e2.estimate(D[s]["bvp"], D[s]["acc"], n[s], p=c)
            mae[(ci, s)] = float(np.nanmean(np.abs(h - D[s]["label"])))
    loso = {}
    for s in D:
        others = [o for o in D if o != s]
        best = min(range(len(configs)), key=lambda ci: np.mean([mae[(ci, o)] for o in others]))
        loso[s] = dict(config=configs[best], heldout_mae=mae[(best, s)])
    final = min(range(len(configs)), key=lambda ci: np.mean([mae[(ci, s)] for s in D]))
    # HR-PPG series used downstream: each subject's E2 with its LOSO-selected config (no self-tuning)
    hp = {s: e2.estimate(D[s]["bvp"], D[s]["acc"], n[s], p=loso[s]["config"]) for s in D}
    hp_e1 = {s: e1.estimate(D[s]["bvp"], D[s]["acc"], n[s])[0] for s in D}
    # ---- C2: B0 with E2 (and E1 for reference)
    b0 = {s: eval_alarm(D[s]["label"], hp[s] > THR) for s in D}
    b0_e1 = {s: eval_alarm(D[s]["label"], hp_e1[s] > THR) for s in D}
    # ---- C3: B2' on negative-window scores (HR-PPG of windows with HR-ECG <= THR), LOSO within dev
    b2p = {}
    for a in (0.05, 0.01, 0.002):
        rows = []; taus = {}
        for s in D:
            cal = np.concatenate([hp[o][D[o]["label"] <= THR] for o in D if o != s])
            tau = stats.conformal_quantile(cal, a); taus[s] = tau
            rows.append(eval_alarm(D[s]["label"], hp[s] > tau))
        b2p[a] = dict(tau=taus, **pooled(rows, [hours[s] for s in D]))
    # ---- C3b: block version of B2' (score = block max of HR-PPG; blocks with any reference event excluded
    #      from calibration; at most one alarm per block), L = 1 min, alpha = target*L/60
    blockres = {}
    for L in (1, 2, 5, 10):
        w = 30 * L
        for target in (0.5, 1, 2):
            a = target * L / 60
            fa_tot = det_tot = ref_tot = 0; H = 0; taus = {}
            for s in D:
                cal = []
                for o in D:
                    if o == s: continue
                    nb = n[o] // w
                    for b in range(nb):
                        sl = slice(b * w, (b + 1) * w)
                        if not (D[o]["label"][sl] > THR).any():
                            cal.append(np.max(hp[o][sl]))
                tau = stats.conformal_quantile(np.array(cal), a); taus[s] = tau
                nb = n[s] // w
                alarm_blocks = [b for b in range(nb) if np.max(hp[s][b * w:(b + 1) * w]) > tau]
                ref = events.episodes(D[s]["label"] > THR, EV["k"], EV["gap"])
                R = [(r0 - EV["tol"], r1 + EV["tol"]) for r0, r1 in (events.to_seconds(r) for r in ref)]
                A = [(b * w * 2, (b + 1) * w * 2 + 6) for b in alarm_blocks]
                ov = lambda x, y: x[0] < y[1] and y[0] < x[1]
                fa_tot += sum(not any(ov(x, r) for r in R) for x in A)
                det_tot += sum(any(ov(x, r) for x in A) for r in R); ref_tot += len(R); H += nb * w * 2 / 3600
            rate, lo, hi = stats.poisson_rate_ci(fa_tot, H)
            blockres[f"L{L}_target{target}"] = dict(alpha_block=a, tau=taus, n_fa=fa_tot, hours=H, far=rate, ci=[lo, hi], det=det_tot, n_ref=ref_tot)
    # ---- C4: lag-1 autocorrelation of block-max scores (E2 HR-PPG and HR-ECG) on dev subjects
    ac = {}
    for L in (1, 2, 5, 10):
        w = 30 * L
        for name, ser in (("HR-PPG(E2)", hp), ("HR-ECG", {s: D[s]["label"] for s in D})):
            vals = []
            for s in D:
                bm = np.array([np.max(ser[s][b * w:(b + 1) * w]) for b in range(n[s] // w)])
                if len(bm) > 3:
                    vals.append(float(np.corrcoef(bm[:-1], bm[1:])[0, 1]))
            ac[f"L{L}|{name}"] = vals
    out = dict(
        reference_published=dict(source="Reiss et al. 2019, Sensors 19(14):3079, 10.3390/s19143079, Table 10 (full text PMC6679242)",
                                 SpaMaPlus=dict(S10=6.17, S11=15.15, S12=12.03, all="11.06 ± 4.8"),
                                 CNN_ensemble=dict(S10=4.03, S11=9.22, S12=9.35, all="7.65 ± 4.2"),
                                 CNN_constrained_all="9.99 ± 5.9 (Table 12)"),
        e2_grid=GRIDS, e2_mae_all={f"{ci}|{s}": v for (ci, s), v in mae.items()}, configs=configs,
        e2_loso=loso, e2_final_config=configs[final],
        e1_mae={s: float(np.nanmean(np.abs(hp_e1[s] - D[s]["label"]))) for s in D},
        b0_e2=dict(per_subject=b0, **pooled(list(b0.values()), [hours[s] for s in D])),
        b0_e1=dict(per_subject=b0_e1, **pooled(list(b0_e1.values()), [hours[s] for s in D])),
        b2prime_e2={str(k): v for k, v in b2p.items()}, b2prime_block_e2=blockres, autocorr_lag1=ac)
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "C_dev.json", "w"), indent=2, default=float)
    print("E2 LOSO:", {s: (v["config"], round(v["heldout_mae"], 2)) for s, v in loso.items()})
    print("E1 MAE:", {s: round(v, 2) for s, v in out["e1_mae"].items()}, "| final E2 config:", configs[final])
    for k in ("b0_e1", "b0_e2"):
        r = out[k]; print(k, f"FA={r['n_fa']} FAR={r['far']:.2f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] det={r['det']}/{r['n_ref']}", [round(x, 2) for x in r["per_subject_far"]])
    for a, r in b2p.items():
        print(f"B2' a={a}: tau={ {s: round(t,1) for s,t in r['tau'].items()} } FA={r['n_fa']} FAR={r['far']:.2f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] det={r['det']}/{r['n_ref']}")
    for k, r in blockres.items():
        print(f"block {k}: a={r['alpha_block']:.4f} tau={ {s: round(t,1) for s,t in r['tau'].items()} } FA={r['n_fa']} FAR={r['far']:.2f} det={r['det']}/{r['n_ref']}")
    for k, v in ac.items():
        print("autocorr", k, [round(x, 2) for x in v])


if __name__ == "__main__":
    main()
