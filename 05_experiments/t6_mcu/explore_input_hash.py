"""T6, EXPLORATORY (development persons e61, u7y, ngh, seed 0; prompt S3d): why does the board's INT8 model input differ
from the PC build of the same C code (h32_core.c) in some windows (input hash board != PC C-core)?

The board returns only a FNV-1a hash of its 1,024-byte INT8 input, not the input itself. This script compiles the
unchanged h32_core.c for the PC with several floating-point contraction settings, recomputes the INT8 inputs of every
window and compares their hashes with the board's. If one PC variant reproduces the board hash on every window, that
variant's inputs stand in for the board's, and the windows whose hash differs from the original PC build (cc -O2,
the build used for qc_<sid>.npy) are characterised by the number of INT8 bins that differ and the largest |difference|.
The firmware object (xtensa-esp32s3-elf-gcc -O2, GNU C default -ffp-contract=fast) contains fused multiply-add
instructions (madd.s / msub.s; checked with objdump, see output). Nothing here changes any analysis code or result.
    uv run python -W ignore 05_experiments/t6_mcu/explore_input_hash.py
Writes 05_experiments/t6_mcu/results/explore_input_hash.json.
"""
import ctypes as C, json, pathlib, subprocess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; SRC = ROOT / "06_firmware/h32_e2/main/h32_core.c"
D = ROOT / "02_data/derived/t6"; OUT = ROOT / "05_experiments/t6_mcu/results"; IDS = ("e61", "u7y", "ngh"); N_S3 = 1800
VARIANTS = {"O2 (as host_check.py)": ["-O2"], "O2 -ffp-contract=off": ["-O2", "-ffp-contract=off"],
            "O2 -ffp-contract=fast": ["-O2", "-ffp-contract=fast"]}
fp = C.POINTER(C.c_float); i8 = C.POINTER(C.c_int8); ptr = lambda a, t: a.ctypes.data_as(t)


def fnv(q):
    x = q.view(np.uint8).astype(np.uint64); h = np.full(len(x), 2166136261, np.uint64)
    for i in range(x.shape[1]):
        h = ((h ^ x[:, i]) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
    return h


def inputs(lib, sid):
    z = np.load(D / f"eval_wild_{sid}.npz"); n = int(z["n"]); ppg = z["ppg64"]; acc = np.ascontiguousarray(z["acc32"])
    X = np.empty((256, 4), np.float32); q = np.empty((n, 1024), np.int8)
    for i in range(n):
        w = np.ascontiguousarray(ppg[128 * i:128 * i + 512]); a = np.ascontiguousarray(acc[64 * i:64 * i + 256])
        lib.h32_spectra(ptr(w, fp), ptr(a, fp), ptr(X, fp)); lib.h32_quant_input(ptr(X, fp), ptr(q[i], i8))
    return q


def char(q0, q1, m):
    d = np.abs(q1[m].astype(int) - q0[m].astype(int)); nb = (d > 0).sum(1)
    return dict(windows=int(m.sum()), bins_differ_median=float(np.median(nb)) if len(nb) else None,
                bins_differ_max=int(nb.max()) if len(nb) else None, bins_differ_total=int(nb.sum()),
                max_abs_int8_diff=int(d.max()) if len(nb) else None)


res = {"label": "exploratory; development persons e61, u7y, ngh; seed 0; PC only, no board run",
       "board_hash_source": "02_data/derived/t6/board_<sid>.npz (ESP32-D0WD-V3 full A); the S3 hashes equal these on the first 1,800 windows (A_S3_reduced.json)",
       "firmware_fp_instructions": {}, "variants": {}}
od = next((ROOT / "06_firmware/.espressif/tools").rglob("xtensa-esp-elf-objdump"), None)
obj = next((ROOT / "06_firmware/h32_e2/build_s3").rglob("h32_core.c.obj"), None)
if od and obj:
    dis = subprocess.run([str(od), "-d", str(obj)], capture_output=True, text=True).stdout.split()
    res["firmware_fp_instructions"] = {k: dis.count(k) for k in ("madd.s", "msub.s", "mul.s", "add.s", "sub.s")}
Q = {}
for name, flags in VARIANTS.items():
    lib_path = D / f"libh32core_explore_{len(Q)}.dylib"
    subprocess.run(["cc", *flags, "-std=c99", "-shared", "-fPIC", "-o", str(lib_path), str(SRC)], check=True)
    lib = C.CDLL(str(lib_path)); v = {"flags": " ".join(flags), "persons": {}}
    for sid in IDS:
        q = inputs(lib, sid); Q[(name, sid)] = q; hb = np.load(D / f"board_{sid}.npz")["in_hash"].astype(np.uint64)
        n = min(len(q), len(hb)); eq = fnv(q[:n]) == hb[:n]
        v["persons"][sid] = dict(windows=int(n), hash_equal_board=int(eq.sum()), hash_equal_board_first_1800=int(eq[:N_S3].sum()))
    v["reproduces_board_on_all_windows"] = all(p["hash_equal_board"] == p["windows"] for p in v["persons"].values())
    res["variants"][name] = v; print(name, json.dumps(v["persons"]))
base = "O2 (as host_check.py)"
for sid in IDS:                                   # the baseline variant must reproduce the stored qc_<sid>.npy
    assert np.array_equal(Q[(base, sid)], np.load(D / f"qc_{sid}.npy")), sid
match = [k for k, v in res["variants"].items() if v["reproduces_board_on_all_windows"]]
res["variant_reproducing_board"] = match[0] if match else None

# Step 2 (no PC variant reproduced the board): recover the board input from its hash. For each window whose board hash
# differs from the PC build, take the K bins whose scaled PC value x = spec / scale lies closest to a rounding boundary
# (|frac(x) - 0.5| smallest), move them one step to the other side (subsets of size 1..3) and recompute the hash.
# A candidate whose hash equals the board hash is taken as the board input (chance match ~ candidates / 2^32).
from itertools import combinations
K = 16; INV = np.float32(1.0 / 0.003921568859368563); ZP = -128
lib = C.CDLL(str(D / "libh32core.dylib"))


def scaled(sid, idx):
    z = np.load(D / f"eval_wild_{sid}.npz"); ppg = z["ppg64"]; acc = np.ascontiguousarray(z["acc32"])
    X = np.empty((256, 4), np.float32); out = np.empty((len(idx), 1024), np.float32)
    for r, i in enumerate(idx):
        w = np.ascontiguousarray(ppg[128 * i:128 * i + 512]); a = np.ascontiguousarray(acc[64 * i:64 * i + 256])
        lib.h32_spectra(ptr(w, fp), ptr(a, fp), ptr(X, fp)); out[r] = X.reshape(-1) * INV
    return out


subsets = [c for k in (1, 2, 3) for c in combinations(range(K), k)]
rec = {}
for sid in IDS:
    q0 = Q[(base, sid)]; hb = np.load(D / f"board_{sid}.npz")["in_hash"].astype(np.uint64); n = min(len(q0), len(hb))
    bad = np.where(fnv(q0[:n]) != hb[:n])[0]; xs = scaled(sid, bad)
    found = np.zeros(len(bad), bool); nb = np.zeros(len(bad), int); dist = []; dist_ok = []
    for r, i in enumerate(bad):
        x = xs[r]; q = q0[i].astype(int)
        assert np.array_equal(np.clip(np.rint(x).astype(int) + ZP, -128, 127), q)
        fr = np.abs(x - np.floor(x) - 0.5); free = (q > -128) & (q < 127); fr = np.where(free, fr, np.inf)
        top = np.argsort(fr)[:K]; step = np.where(x[top] - np.rint(x[top]) > 0, 1, -1)     # other side of the boundary
        cand = np.repeat(q0[i][None, :], len(subsets), 0).astype(np.int16)
        for c, s in enumerate(subsets):
            cand[c, top[list(s)]] += step[list(s)]
        h = fnv(cand.astype(np.int8)); hit = np.where(h == hb[i])[0]
        if len(hit):
            s = subsets[hit[0]]; found[r] = True; nb[r] = len(s); dist_ok += [float(fr[top[j]]) for j in s]          # distance from the rounding boundary, in quantisation steps
    first = bad < N_S3
    rec[sid] = dict(windows_hash_differs=int(len(bad)), recovered=int(found.sum()),
                    recovered_first_1800=f"{int(found[first].sum())}/{int(first.sum())}",
                    bins_differ_counts={str(k): int(np.sum(nb[found] == k)) for k in (1, 2, 3)},
                    bins_differ_counts_first_1800={str(k): int(np.sum(nb[found & first] == k)) for k in (1, 2, 3)},
                    max_abs_int8_diff=1 if found.any() else None,
                    max_distance_of_changed_value_from_boundary=float(max(dist_ok)) if dist_ok else None)
    print(sid, json.dumps(rec[sid]))
res["recovery_from_hash"] = dict(method=f"flip 1-3 of the {K} bins closest to a rounding boundary by one step; accept on equal FNV-1a hash",
                                 candidates_per_window=len(subsets), persons=rec)
for p in D.glob("libh32core_explore_*.dylib"):
    p.unlink()
print("variant match:", res["variant_reproducing_board"], res["firmware_fp_instructions"])
json.dump(res, open(OUT / "explore_input_hash.json", "w"), indent=1)
