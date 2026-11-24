"""T6 step 2 host check: the firmware's C core (06_firmware/h32_e2/main/h32_core.c, unchanged) compiled for the PC and
compared with the frozen Python pipeline on the dev persons (seed 0). The SpecFCN INT8 output is taken from the
TFLite reference kernels (quantize.py), so this isolates the C pre-/post-processing and the C block/rule code.
    uv run python -W ignore 05_experiments/t6_mcu/host_check.py
"""
import ctypes as C, json, pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import confirm, method as M  # noqa: E402

SRC = ROOT / "06_firmware/h32_e2/main"; D = ROOT / "02_data/derived/t6"; LIB = D / "libh32core.dylib"
subprocess.run(["cc", "-O2", "-std=c99", "-shared", "-fPIC", "-o", str(LIB), str(SRC / "h32_core.c")], check=True)
L = C.CDLL(str(LIB)); fp = C.POINTER(C.c_float); i8 = C.POINTER(C.c_int8)


class Blk(C.Structure):
    _fields_ = [("state", C.c_uint8), ("j", C.c_uint16), ("S", C.c_float), ("tauK", C.c_float), ("tauR", C.c_float),
                ("hasK", C.c_uint8), ("decK", C.c_uint8), ("hasR", C.c_uint8), ("decR", C.c_uint8)]


HMM = C.c_char * (256 * 4 + 8); RULES = C.c_char * 1024
L.h32_rules_step.restype = C.c_int; L.h32_acc_state.restype = C.c_int
ptr = lambda a, t: a.ctypes.data_as(t)
meta = json.load(open(ROOT / "05_experiments/t6_mcu/model/quant_meta.json")); si, zi = meta["input"]["scale"], meta["input"]["zero_point"]
so, zo = meta["output"]["scale"], meta["output"]["zero_point"]


def py_track_int8(q):
    """eval_int8_pc.track on the dequantised int8 logits (float64 softmax)."""
    from h32 import e2cnn
    lg = ((q.astype(np.float32) - zo) * so)[:, e2cnn.FCN_MASK].astype(np.float64)
    lg -= lg.max(1, keepdims=True); P = np.exp(lg); P = (P / P.sum(1, keepdims=True)).astype(np.float32)
    return P.max(1), e2cnn.hmm_track_grid(P, e2cnn.FCN_GRID, 3.0).astype(np.float32)


res = {}
for sid in ("e61", "u7y", "ngh"):
    z = np.load(D / f"eval_wild_{sid}.npz"); n = int(z["n"]); ppg = z["ppg64"]; acc = np.ascontiguousarray(z["acc32"])
    Xpy = np.transpose(z["X"], (0, 2, 1))                     # (n, 256, 4)
    Xc = np.empty((n, 256, 4), np.float32); st_c = np.empty(n, int); qc = np.empty((n, 1024), np.int8)
    for i in range(n):
        w = np.ascontiguousarray(ppg[128 * i:128 * i + 512]); a = np.ascontiguousarray(acc[64 * i:64 * i + 256])
        L.h32_spectra(ptr(w, fp), ptr(a, fp), ptr(Xc[i], fp)); L.h32_quant_input(ptr(Xc[i], fp), ptr(qc[i], i8))
        st_c[i] = L.h32_acc_state(ptr(a, fp))
    np.save(D / f"qc_{sid}.npy", qc); np.save(D / f"stc_{sid}.npy", st_c)   # C-side model inputs (board emulation)
    qpy = np.clip(np.round(Xpy.reshape(n, -1) / si) + zi, -128, 127).astype(np.int8)
    st_py = M.states(acc.astype(np.float64), n)
    r = dict(n=n, spectra_max_abs=float(np.max(np.abs(Xc - Xpy))), spectra_p99_abs=float(np.quantile(np.abs(Xc - Xpy), 0.99)),
             quant_input_values_differ=int(np.sum(qc != qpy)), quant_input_windows_differ=int(np.sum(np.any(qc != qpy, 1))),
             state_differs=int(np.sum(st_c != st_py)))
    # post-processing on the same int8 logits (TFLite reference kernels, from quantize.py)
    q8 = np.load(D / f"logit_int8q_wild_{sid}.npy"); sq_py, hp_py = py_track_int8(q8)
    h = HMM(); L.h32_hmm_init(h); hr = C.c_float(); sq = C.c_float(); hp_c = np.empty(n, np.float32); sq_c = np.empty(n, np.float32)
    for i in range(n):
        row = np.ascontiguousarray(q8[i]); L.h32_post(h, ptr(row, i8), C.byref(hr), C.byref(sq)); hp_c[i] = hr.value; sq_c[i] = sq.value
    r.update(post_dHR_max=float(np.max(np.abs(hp_c - hp_py))), post_dHR_windows=int(np.sum(hp_c != hp_py)),
             post_dSQI_max=float(np.max(np.abs(sq_c - sq_py))),
             post_mon_differs=int(np.sum((sq_c >= confirm.TH) != (sq_py >= confirm.TH))))
    # block machinery and rules on the Python float stream (confirm cache, seed 0)
    cz = np.load(ROOT / f"02_data/derived/confirm_cache/dryrun_dev_r8_1/{sid}_seed0.npz")
    stv, sqv, hpv = cz["st"], cz["sq"], cz["hp3"]; mon = sqv >= confirm.TH
    cl = confirm.stream(stv, sqv, hpv); S = [cl["S"][k][:cl["nc"][k]].astype(float) for k in range(3)]
    tK = [confirm.block_rule(S[k], len(S[k]), "K") for k in range(3)]; tR = [confirm.block_rule(S[k], len(S[k]), "R") for k in range(3)]
    rl = RULES(); L.h32_rules_init(rl); b = Blk(); got = []
    for i in range(len(stv)):
        if L.h32_rules_step(rl, int(stv[i]), int(mon[i]), C.c_float(float(hpv[i])), C.byref(b)):
            got.append((b.state, b.j, b.S, b.hasK, b.tauK, b.decK, b.hasR, b.tauR, b.decR))
    nblk = sum(len(s) for s in S); bad = 0; badK = badR = 0
    for (k, j, Sc, hk, tk, dk, hr_, trr, dr) in got:
        if Sc != S[k][j] and not (np.isinf(Sc) and np.isinf(S[k][j])): bad += 1
        if bool(hk) != (j in tK[k]) or (hk and (tk != tK[k][j] or bool(dk) != (S[k][j] > tK[k][j]))): badK += 1
        if bool(hr_) != (j in tR[k]) or (hr_ and (trr != tR[k][j] or bool(dr) != (S[k][j] > tR[k][j]))): badR += 1
    r.update(rules_blocks_python=nblk, rules_blocks_c=len(got), rules_S_differ=bad, rules_K_differ=badK, rules_R_differ=badR)
    res[sid] = r; print(sid, r)
json.dump(res, open(ROOT / "05_experiments/t6_mcu/results/host_check.json", "w"), indent=1)
