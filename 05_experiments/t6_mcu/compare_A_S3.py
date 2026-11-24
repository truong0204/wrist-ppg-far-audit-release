"""T6 on the ESP32-S3, step A-S3-1 (PLAN_T6 S3 plan item 4, amendment 4): first 1,800 windows of each dev person,
both models, compared per window with
  (1) the ESP32-D0WD-V3 dense Measurement A outputs (board_<sid>.npz): input hash, int8 logits, HR, SQI, state.
      Stop condition of the reduced A (dense model): all 5,400 windows byte-identical in these five fields;
  (2) the PC INT8 reference on C-core inputs (logit_emul_<sid>.npy), overall and on windows with equal input hash;
  (3) the PC float pipeline (amendment 4): float logits (eval_wild_<sid>.npz, PyTorch, masked 194 bins) against the
      dequantised board logits; HR, SQI flag and state against the float confirmatory dry-run cache (seed 0).
No block or K/R statistic: 1,800 windows hold no complete test block (S3 plan item 4).
    uv run python -W ignore 05_experiments/t6_mcu/compare_A_S3.py
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import confirm, e2cnn  # noqa: E402

D = ROOT / "02_data/derived/t6"; OUT = ROOT / "05_experiments/t6_mcu/results"; N = 1800; IDS = ("e61", "u7y", "ngh")
SCALE, ZP = json.load(open(ROOT / "05_experiments/t6_mcu/model/quant_meta_dense.json"))["output_quantization"]
MASK = np.asarray(e2cnn.FCN_MASK, bool); TH = confirm.TH
FIELDS = ("in_hash", "logits", "hr", "sqi", "st")


def same(a, b):
    return np.all(a == b, axis=1) if a.ndim > 1 else (a == b)


def q(v):
    v = np.abs(np.asarray(v, float)); return dict(median=float(np.median(v)), p99=float(np.quantile(v, 0.99)), max=float(v.max()))


def fnv(qc):
    x = qc.view(np.uint8).astype(np.uint64); h = np.full(len(x), 2166136261, np.uint64)
    for i in range(x.shape[1]):
        h = ((h ^ x[:, i]) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
    return h


res = {"windows_per_person": N, "models": {}}
for model in ("dense", "orig"):
    M = {"persons": {}}
    for sid in IDS:
        s = np.load(D / f"s3A_{model}_{sid}.npz"); e = np.load(D / f"board_{sid}.npz")
        n = min(N, len(s["hr"])); r = {"windows": int(n), "resends": int(s["resends"]) if "resends" in s.files else None}
        r["vs_esp32"] = {k: int(same(s[k][:n], e[k][:n]).sum()) for k in FIELDS}
        r["vs_esp32_all_five"] = int(np.all([same(s[k][:n], e[k][:n]) for k in FIELDS], axis=0).sum())
        lg = s["logits"][:n]; e1 = np.load(D / f"logit_emul_{sid}.npy")[:n]
        eqh = s["in_hash"][:n].astype(np.uint64) == fnv(np.load(D / f"qc_{sid}.npy")[:n])
        r["vs_pc_int8"] = dict(input_hash_equal=int(eqh.sum()), logits_identical=int(same(lg, e1).sum()),
                              logits_identical_given_equal_hash=f"{int(same(lg[eqh], e1[eqh]).sum())}/{int(eqh.sum())}")
        lf = np.load(D / f"eval_wild_{sid}.npz")["logit_float"][:n]
        ld = (lg.astype(np.float32) - ZP) * SCALE; ld = ld[:, MASK]
        c = np.load(ROOT / f"02_data/derived/confirm_cache/dryrun_dev_r8_1/{sid}_seed0.npz")
        hr_d, sq_d, st_d = s["hr"][:n].astype(np.float32), s["sqi"][:n].astype(np.float32), s["st"][:n].astype(int)
        r["vs_pc_float"] = dict(dlogit=q(ld - lf), argmax_equal=int(np.sum(ld.argmax(1) == lf.argmax(1))),
                                dHR=q(hr_d - c["hp3"][:n]), monitored_flag_differs=int(np.sum((sq_d >= TH) != (c["sq"][:n] >= TH))),
                                state_differs=int(np.sum(st_d != c["st"][:n])))
        M["persons"][sid] = r
    tot = sum(M["persons"][p]["vs_esp32_all_five"] for p in IDS); nn = sum(M["persons"][p]["windows"] for p in IDS)
    M["vs_esp32_all_five_total"] = f"{tot}/{nn}"
    if model == "dense":
        M["reduced_A_stop_condition_met"] = bool(tot == nn)
    res["models"][model] = M
# the two models on the S3 against each other (their INT8 outputs are identical on the PC, amendment 1)
res["orig_vs_dense_on_s3"] = {sid: {k: int(same(np.load(D / f"s3A_orig_{sid}.npz")[k][:N], np.load(D / f"s3A_dense_{sid}.npz")[k][:N]).sum())
                                    for k in FIELDS} for sid in IDS}
json.dump(res, open(OUT / "A_S3_reduced.json", "w"), indent=1)
print(json.dumps({m: {"all_five": v["vs_esp32_all_five_total"], "stop": v.get("reduced_A_stop_condition_met")} for m, v in res["models"].items()}))
