"""T6 board emulation on the PC: C core post-processing and rules (h32_core.c, compiled by host_check.py) on the INT8 outputs
of emulate_tflm.py, with the C-side ACC states. Writes device-like outputs 02_data/derived/t6/emul_<sid>.npz for compare_A.py.
This is a prediction of Measurement A, not a measurement.
    uv run python -W ignore 05_experiments/t6_mcu/board_emul.py
"""
import ctypes as C, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "02_data/derived/t6"
L = C.CDLL(str(D / "libh32core.dylib")); i8 = C.POINTER(C.c_int8)
TH = 0.02896360121667385


class Blk(C.Structure):
    _fields_ = [("state", C.c_uint8), ("j", C.c_uint16), ("S", C.c_float), ("tauK", C.c_float), ("tauR", C.c_float),
                ("hasK", C.c_uint8), ("decK", C.c_uint8), ("hasR", C.c_uint8), ("decR", C.c_uint8)]


for sid in ("e61", "u7y", "ngh"):
    q8 = np.load(D / f"logit_emul_{sid}.npy"); st = np.load(D / f"stc_{sid}.npy"); n = len(q8)
    h = (C.c_char * 1100)(); L.h32_hmm_init(h); rl = (C.c_char * 1024)(); L.h32_rules_init(rl)
    hr = C.c_float(); sq = C.c_float(); b = Blk(); HR = np.empty(n, np.float32); SQ = np.empty(n, np.float32); blk = []
    for i in range(n):
        row = np.ascontiguousarray(q8[i]); L.h32_post(h, row.ctypes.data_as(i8), C.byref(hr), C.byref(sq))
        HR[i], SQ[i] = hr.value, sq.value
        if L.h32_rules_step(rl, int(st[i]), int(sq.value >= TH), C.c_float(hr.value), C.byref(b)):
            blk.append((b.state, b.j, b.S, b.hasK, b.tauK, b.decK, b.hasR, b.tauR, b.decR))
    np.savez(D / f"emul_{sid}.npz", hr=HR, sqi=SQ, st=st, blk=np.array(blk, float)); print(sid, n, len(blk))
