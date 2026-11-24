"""T6 step 1b (separate env 05_experiments/t6_mcu/.venv, TensorFlow 2.21.0): SpecFCN seed 0 -> Keras (float,
weights copied) -> full-integer INT8 TFLite (PLAN_T6 §1). Runs the INT8 model with the TFLite interpreter on the
PC for every evaluation set exported by export_inputs.py.
    05_experiments/t6_mcu/.venv/bin/python 05_experiments/t6_mcu/quantize.py
Docs followed (read 2026-11-23): developers.google.com/edge/litert/models/post_training_integer_quant and
.../post_training_quantization (integer-only settings; in-converter PTQ marked deprecated there).
"""
import json, pathlib
import numpy as np, tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "02_data/derived/t6"; OUTM = ROOT / "05_experiments/t6_mcu/model"; OUTM.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(0)
W = np.load(D / "weights_seed0.npz"); mask = W["mask"].astype(bool)

inp = tf.keras.Input((256, 4), batch_size=1, name="spectra")   # fixed batch 1, as on the board
x = inp
for i, (li, dil, act) in enumerate([(0, 1, "relu"), (2, 1, "relu"), (4, 2, "relu"), (6, 1, None)]):
    w = W[f"net.{li}.weight"]                                     # torch (out, in, k)
    conv = tf.keras.layers.Conv1D(w.shape[0], w.shape[2], padding="same", dilation_rate=dil, activation=act, name=f"conv{i}")
    x = conv(x); conv.set_weights([np.transpose(w, (2, 1, 0)), W[f"net.{li}.bias"]])
out = tf.keras.layers.Reshape((256,), name="logits")(x)
model = tf.keras.Model(inp, out)

to_k = lambda X: np.ascontiguousarray(np.transpose(X, (0, 2, 1)))   # (N, 4, 256) -> (N, 256, 4)
sets = {f"dalia_{s}": D / f"eval_dalia_{s}.npz" for s in ("S10", "S11", "S12")}
sets.update({f"wild_{s}": D / f"eval_wild_{s}.npz" for s in ("e61", "u7y", "ngh")})

q0 = {}
for name, f in sets.items():
    z = np.load(f); lk = model.predict(to_k(z["X"]), batch_size=2048, verbose=0)[:, mask]
    q0[name] = float(np.max(np.abs(lk - z["logit_float"])))
print("Q0 max |logit keras - torch|:", q0)

calib = to_k(np.load(D / "calib_dalia_500.npy"))
def rep():
    for i in range(len(calib)):
        yield [calib[i:i + 1].astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
(OUTM / "specfcn_seed0_int8.tflite").write_bytes(tfl)
fl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
(OUTM / "specfcn_seed0_float.tflite").write_bytes(fl)

REF = tf.lite.experimental.OpResolverType.BUILTIN_REF          # reference kernels (closest to TFLM), not XNNPACK
it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=REF); it.allocate_tensors()
di, do = it.get_input_details()[0], it.get_output_details()[0]
si, zi = di["quantization"]; so, zo = do["quantization"]
ops = sorted({d["op_name"] for d in it._get_ops_details()})
meta = dict(tf_version=tf.__version__, tflite_bytes=len(tfl), float_tflite_bytes=len(fl), ops=ops,
            input=dict(shape=di["shape"].tolist(), dtype=str(di["dtype"]), scale=si, zero_point=zi),
            output=dict(shape=do["shape"].tolist(), dtype=str(do["dtype"]), scale=so, zero_point=zo),
            q0_max_abs_logit_keras_vs_torch=q0, calib_windows=len(calib))
print(json.dumps(meta, indent=1))
for name, f in sets.items():                                    # one window per invoke, as on the board
    z = np.load(f); X = to_k(z["X"]); out = np.empty((len(X), 256), np.int8)
    for i in range(len(X)):
        it.set_tensor(di["index"], np.clip(np.round(X[i:i + 1] / si) + zi, -128, 127).astype(np.int8)); it.invoke()
        out[i] = it.get_tensor(do["index"])[0]
    np.save(D / f"logit_int8q_{name}.npy", out)                  # raw int8 (256 bins)
    np.save(D / f"logit_int8_{name}.npy", ((out.astype(np.float32) - zo) * so)[:, mask])
json.dump(meta, open(ROOT / "05_experiments/t6_mcu/model/quant_meta.json", "w"), indent=1)
print("done")
