"""T6 step 1c (project environment): INT8 (TFLite reference kernels, PC) vs float (PyTorch), PLAN_T6 §2.
Same float64 spectra on both sides, so only the quantisation of SpecFCN differs. Seed 0. Dev persons only.
    uv run python -W ignore 05_experiments/t6_mcu/eval_int8_pc.py
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import e2cnn, confirm  # noqa: E402

D = ROOT / "02_data/derived/t6"; OUT = ROOT / "05_experiments/t6_mcu/results"; OUT.mkdir(exist_ok=True)
TH = confirm.TH; GRID = e2cnn.FCN_GRID; STEP = float(np.mean(np.diff(GRID))); DIFF_S = 0.41   # PLAN_T6 §3 item 4


def softmax(L):
    L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)


def track(logit):
    P = softmax(logit.astype(np.float64)).astype(np.float32)
    return P.max(1), e2cnn.hmm_track_grid(P, GRID, 3.0).astype(np.float32)


def q(v):
    v = np.abs(np.asarray(v, float)); return dict(median=float(np.median(v)), p99=float(np.quantile(v, 0.99)), max=float(v.max()))


def blocks_and_rules(st, sq, hp):
    cl = confirm.stream(st, sq, hp); S = [cl["S"][k][:cl["nc"][k]].astype(float) for k in range(3)]
    return cl, S, {rule: [confirm.block_rule(S[k], len(S[k]), rule) for k in range(3)] for rule in ("K", "R")}


res = {"theta": TH, "grid_step_bpm": STEP, "block_score_diff_threshold_bpm": DIFF_S, "sets": {}}
for name in ["dalia_S10", "dalia_S11", "dalia_S12", "wild_e61", "wild_u7y", "wild_ngh"]:
    z = np.load(D / f"eval_{name}.npz")
    sq_f, hp_f = track(z["logit_float"]); sq_i, hp_i = track(np.load(D / f"logit_int8_{name}.npy"))
    r = dict(n_windows=int(len(hp_f)), dHR_all=q(hp_i - hp_f), frac_dHR_nonzero=float(np.mean(hp_i != hp_f)),
             dSQI=q(sq_i - sq_f), monitored_flag_differs=int(np.sum((sq_f >= TH) != (sq_i >= TH))))
    both = (sq_f >= TH) & (sq_i >= TH); r["dHR_monitored_both"] = q((hp_i - hp_f)[both]); r["n_monitored_both"] = int(both.sum())
    if name.startswith("dalia"):
        y = z["y"]; r["MAE_vs_ECG_float"] = float(np.mean(np.abs(hp_f - y))); r["MAE_vs_ECG_int8"] = float(np.mean(np.abs(hp_i - y)))
    else:
        sid = name.split("_")[1]
        c = np.load(ROOT / f"02_data/derived/confirm_cache/dryrun_dev_r8_1/{sid}_seed0.npz")
        r["check_float_vs_confirm_cache"] = dict(max_dSQI=float(np.max(np.abs(sq_f - c["sq"]))), max_dHR=float(np.max(np.abs(hp_f - c["hp3"]))),
                                                  state_equal=bool(np.array_equal(z["st"], c["st"])))
        st = c["st"]
        _, Sf, Rf = blocks_and_rules(st, sq_f, hp_f); _, Si, Ri = blocks_and_rules(st, sq_i, hp_i)
        blk = dict(compared=0, differ=0, steps={"0": 0, "1": 0, "2": 0, ">=3": 0}, only_one_side=0)
        for k in range(3):
            n = min(len(Sf[k]), len(Si[k])); blk["only_one_side"] += abs(len(Sf[k]) - len(Si[k]))
            d = np.abs(Si[k][:n] - Sf[k][:n]); fin = np.isfinite(Sf[k][:n]) & np.isfinite(Si[k][:n])
            d = np.where(fin, d, np.where(np.isfinite(Sf[k][:n]) == np.isfinite(Si[k][:n]), 0.0, np.inf))
            blk["compared"] += n; blk["differ"] += int(np.sum(d > DIFF_S))
            st_ = np.round(d / STEP)
            for lab, m_ in (("0", st_ == 0), ("1", st_ == 1), ("2", st_ == 2), (">=3", st_ >= 3)):
                blk["steps"][lab] += int(np.sum(m_))
        r["blocks"] = blk
        dec = {}
        for rule in ("K", "R"):
            tot = dif = one = ntau = 0; dtau = 0.0; ktau = {}
            for k in range(3):
                tf_, ti_ = Rf[rule][k], Ri[rule][k]
                for j in tf_:
                    if j not in ti_:
                        one += 1; continue
                    tot += 1; dif += int((Sf[k][j] > tf_[j]) != (Si[k][j] > ti_[j]))
                    if tf_[j] != ti_[j]:
                        ntau += 1; dtau = max(dtau, abs(ti_[j] - tf_[j]))
                one += sum(1 for j in ti_ if j not in tf_)
                if rule == "K" and tf_ and ti_:
                    ktau[str(k)] = dict(float=next(iter(tf_.values())), int8=next(iter(ti_.values())))
            dec[rule] = dict(test_blocks=tot, decisions_differ=dif, test_blocks_one_side_only=one, tau_differs=ntau, max_abs_dtau=dtau)
            if rule == "K":
                dec[rule]["tau_per_state"] = ktau
        r["decisions"] = dec
    res["sets"][name] = r
    print(name, json.dumps(r)[:400])
json.dump(res, open(OUT / "int8_vs_float_pc.json", "w"), indent=1)
