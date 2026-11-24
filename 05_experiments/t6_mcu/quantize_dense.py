"""T6 amendment 1 (PI 23-11): board model = SpecFCN seed 0 with the dilated layer (k = 7, dilation 2) written as a dense
conv (k = 13, zero taps at odd positions), because ESP-NN does not accelerate dilated conv. Same TF 2.21.0 full-int8 PTQ
and DaLiA calibration set as quantize.py. Checks that the INT8 logits equal those of the original INT8 model.
    05_experiments/t6_mcu/.venv/bin/python 05_experiments/t6_mcu/quantize_dense.py
"""
import json
import pathlib, numpy as np, tensorflow as tf
ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "02_data/derived/t6"
W = np.load(D / "weights_seed0.npz"); mask = W["mask"].astype(bool)
inp = tf.keras.Input((256, 4), batch_size=1); x = inp
for i, (li, act) in enumerate([(0, "relu"), (2, "relu"), (4, "relu"), (6, None)]):
    w = np.transpose(W[f"net.{li}.weight"], (2, 1, 0))
    if li == 4:
        w13 = np.zeros((13,) + w.shape[1:], w.dtype); w13[0::2] = w; w = w13
    c = tf.keras.layers.Conv1D(w.shape[2], w.shape[0], padding="same", activation=act); x = c(x); c.set_weights([w, W[f"net.{li}.bias"]])
m = tf.keras.Model(inp, tf.keras.layers.Reshape((256,))(x))
calib = np.ascontiguousarray(np.transpose(np.load(D / "calib_dalia_500.npy"), (0, 2, 1)))
cv = tf.lite.TFLiteConverter.from_keras_model(m); cv.optimizations = [tf.lite.Optimize.DEFAULT]
cv.representative_dataset = lambda: ([calib[i:i+1]] for i in range(len(calib)))
cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; cv.inference_input_type = tf.int8; cv.inference_output_type = tf.int8
tfl = cv.convert(); (ROOT / "05_experiments/t6_mcu/model/specfcn_seed0_int8_dense.tflite").write_bytes(tfl)
it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF); it.allocate_tensors()
di, do = it.get_input_details()[0], it.get_output_details()[0]; print("ops", sorted({d["op_name"] for d in it._get_ops_details()}), "bytes", len(tfl), "out q", do["quantization"])
res = dict(tf_version=tf.__version__, tflite_bytes=len(tfl), ops=sorted({d["op_name"] for d in it._get_ops_details()}),
           output_quantization=list(do["quantization"]), identical_windows={})
si, zi = di["quantization"]
for sid in ("e61", "u7y", "ngh"):
    qc = np.load(D / f"qc_{sid}.npy"); ref = np.load(D / f"logit_emul_{sid}.npy"); diff = 0        # C-core inputs
    for i in range(len(qc)):
        it.set_tensor(di["index"], qc[i].reshape(1, 256, 4)); it.invoke(); diff += int(np.any(it.get_tensor(do["index"])[0] != ref[i]))
    X = np.ascontiguousarray(np.transpose(np.load(D / f"eval_wild_{sid}.npz")["X"], (0, 2, 1))); ref2 = np.load(D / f"logit_int8q_wild_{sid}.npy"); diff2 = 0
    for i in range(len(X)):                                                                       # Python-feature inputs
        it.set_tensor(di["index"], np.clip(np.round(X[i:i + 1] / si) + zi, -128, 127).astype(np.int8)); it.invoke()
        diff2 += int(np.any(it.get_tensor(do["index"])[0] != ref2[i]))
    res["identical_windows"][sid] = dict(n=len(qc), c_core_inputs=len(qc) - diff, python_inputs=len(X) - diff2)
    print(sid, res["identical_windows"][sid])
json.dump(res, open(ROOT / "05_experiments/t6_mcu/model/quant_meta_dense.json", "w"), indent=1)
