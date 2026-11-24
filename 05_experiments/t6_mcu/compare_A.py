"""T6 Measurement A (PLAN_T6 §3): device outputs vs the float PC pipeline (confirmatory dry-run cache, seed 0), dev persons.
Device outputs: 02_data/derived/t6/<source>_<sid>.npz with per-window hr, sqi, st and the device's own block records
(blk: state, j, S, hasK, tauK, decK, hasR, tauR, decR).
    uv run python -W ignore 05_experiments/t6_mcu/compare_A.py <source>     # source = emul (PC emulation) or board
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import confirm, e2cnn  # noqa: E402

SRC = sys.argv[1]; D = ROOT / "02_data/derived/t6"; OUT = ROOT / "05_experiments/t6_mcu/results"
TH = confirm.TH; STEP = float(np.mean(np.diff(e2cnn.FCN_GRID))); DIFF_S = 0.41


def q(v):
    v = np.abs(np.asarray(v, float))
    return dict(median=float(np.median(v)), p99=float(np.quantile(v, 0.99)), max=float(v.max())) if len(v) else None


def py_blocks(st, sq, hp):
    cl = confirm.stream(st, sq, hp); S = [cl["S"][k][:cl["nc"][k]].astype(float) for k in range(3)]
    return S, {r: [confirm.block_rule(S[k], len(S[k]), r) for k in range(3)] for r in ("K", "R")}


res = {"source": SRC, "threshold_block_score_bpm": DIFF_S, "persons": {}}
for sid in ("e61", "u7y", "ngh"):
    f = np.load(ROOT / f"02_data/derived/confirm_cache/dryrun_dev_r8_1/{sid}_seed0.npz"); d = np.load(D / f"{SRC}_{sid}.npz")
    n = min(len(f["hp3"]), len(d["hr"])); st_f, sq_f, hp_f = f["st"][:n], f["sq"][:n], f["hp3"][:n]
    st_d, sq_d, hp_d = d["st"][:n].astype(int), d["sqi"][:n].astype(np.float32), d["hr"][:n].astype(np.float32)
    mf, md = sq_f >= TH, sq_d >= TH
    r = dict(n_windows_float=int(len(f["hp3"])), n_windows_device=int(len(d["hr"])),
             dHR_all=q(hp_d - hp_f), dHR_monitored_both=q((hp_d - hp_f)[mf & md]), frac_dHR_nonzero=float(np.mean(hp_d != hp_f)),
             monitored_flag_differs=int(np.sum(mf != md)), state_differs=int(np.sum(st_f != st_d)))
    Sf, Rf = py_blocks(st_f, sq_f, hp_f); Sd, Rd = py_blocks(st_d, sq_d, hp_d)
    # device's own block code vs the Python block code on the device's per-window outputs
    blk = d["blk"]; own_bad = 0
    if SRC != "emul":     # board records (replay.py, struct h32_block_t): state, j, S, tauK, tauR, hasK, decK, hasR, decR
        blk = blk[:, [0, 1, 2, 5, 3, 6, 7, 4, 8]]          # -> emulation order: state, j, S, hasK, tauK, decK, hasR, tauR, decR
    for (k, j, S, hk, tk, dk, hr_, tr, dr) in blk:
        k, j = int(k), int(j); py_S = Sd[k][j] if j < len(Sd[k]) else None
        okS = py_S is not None and (S == py_S or (np.isinf(S) and np.isinf(py_S)))
        okK = bool(hk) == (j in Rd["K"][k]) and (not hk or (tk == Rd["K"][k][j] and bool(dk) == (py_S > tk)))
        okR = bool(hr_) == (j in Rd["R"][k]) and (not hr_ or (tr == Rd["R"][k][j] and bool(dr) == (py_S > tr)))
        own_bad += int(not (okS and okK and okR))
    r["device_block_code_vs_python_on_device_outputs"] = dict(records=int(len(blk)), python_blocks=int(sum(len(s) for s in Sd)), mismatches=own_bad)
    b = dict(compared=0, differ=0, steps={"0": 0, "1": 0, "2": 0, ">=3": 0}, only_one_side=0)
    for k in range(3):
        m = min(len(Sf[k]), len(Sd[k])); b["only_one_side"] += abs(len(Sf[k]) - len(Sd[k]))
        a, c = Sf[k][:m], Sd[k][:m]; both_inf = np.isinf(a) & np.isinf(c)
        dd = np.where(both_inf, 0.0, np.abs(c - a)); b["compared"] += m; b["differ"] += int(np.sum(dd > DIFF_S))
        s_ = np.round(dd / STEP)
        for lab, mm in (("0", s_ == 0), ("1", s_ == 1), ("2", s_ == 2), (">=3", s_ >= 3)):
            b["steps"][lab] += int(np.sum(mm))
    r["blocks"] = b
    for rule in ("K", "R"):
        tot = dif = one = ntau = 0; mx = 0.0; per = {}
        for k in range(3):
            tf_, td = Rf[rule][k], Rd[rule][k]
            for j, t in tf_.items():
                if j not in td:
                    one += 1; continue
                tot += 1; dif += int((Sf[k][j] > t) != (Sd[k][j] > td[j]))
                if t != td[j]:
                    ntau += 1; mx = max(mx, abs(td[j] - t))
            one += sum(1 for j in td if j not in tf_)
            if rule == "K" and tf_ and td:
                per[str(k)] = dict(float=float(next(iter(tf_.values()))), device=float(next(iter(td.values()))))
        r[rule] = dict(test_blocks=tot, decisions_differ=dif, test_blocks_one_side_only=one, tau_differs=ntau, max_abs_dtau=mx)
        if rule == "K":
            r[rule]["tau_per_state"] = per
    if "logits" in d.files:                                    # amendment 1 (b)
        lg = d["logits"][:n]; e1 = np.load(D / f"logit_emul_{sid}.npy")[:n]; e2 = np.load(D / f"logit_int8q_wild_{sid}.npy")[:n]
        qc = np.load(D / f"qc_{sid}.npy")[:n].view(np.uint8).astype(np.uint64); h = np.full(n, 2166136261, np.uint64)
        for i in range(qc.shape[1]):
            h = ((h ^ qc[:, i]) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
        r["logits_bit_identical_vs_orig_pc_ccore_inputs"] = int(np.sum(np.all(lg == e1, 1)))
        r["logits_bit_identical_vs_orig_pc_python_inputs"] = int(np.sum(np.all(lg == e2, 1)))
        r["input_hash_equal_board_vs_pc_ccore"] = int(np.sum(d["in_hash"][:n].astype(np.uint64) == h))
        eq = d["in_hash"][:n].astype(np.uint64) == h                    # board inference vs PC inference on identical inputs
        r["logits_bit_identical_given_equal_input_hash"] = f"{int(np.sum(np.all(lg[eq] == e1[eq], 1)))}/{int(eq.sum())}"
        r["resends"] = int(d["resends"]) if "resends" in d.files else None
    res["persons"][sid] = r; print(sid, json.dumps({k: r[k] for k in ("monitored_flag_differs", "blocks", "K", "R")})[:500])
tot = {rule: [sum(res["persons"][s][rule][x] for s in res["persons"]) for x in ("decisions_differ", "test_blocks")] for rule in ("K", "R")}
res["pooled"] = {rule: dict(decisions_differ=v[0], test_blocks=v[1]) for rule, v in tot.items()}
print("pooled", res["pooled"])
rep = D / f"{SRC}_e61_rep2.npz"
if rep.exists():                                               # determinism: e61 replayed twice
    x, y = np.load(D / f"{SRC}_e61.npz"), np.load(rep)
    # NaN marks "no threshold" fields in block records; NaN at the same position counts as equal
    res["determinism_e61"] = {k: bool(np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f"))
                              for k in ("hr", "sqi", "st", "blk", "logits", "in_hash")}
    print("determinism", res["determinism_e61"])
json.dump(res, open(OUT / f"A_{SRC}.json", "w"), indent=1)
