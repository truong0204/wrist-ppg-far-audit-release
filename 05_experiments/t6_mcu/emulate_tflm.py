"""T6 board emulation, TF env: run the INT8 model (TFLite reference kernels) on the model inputs quantised by the C core
(qc_<sid>.npy from host_check.py). Predicts the board's model output if TFLM gives the reference-kernel result.
    05_experiments/t6_mcu/.venv/bin/python 05_experiments/t6_mcu/emulate_tflm.py
"""
import pathlib
import numpy as np, tensorflow as tf
ROOT = pathlib.Path(__file__).resolve().parents[2]; D = ROOT / "02_data/derived/t6"
it = tf.lite.Interpreter(model_path=str(ROOT / "05_experiments/t6_mcu/model/specfcn_seed0_int8.tflite"),
                         experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
it.allocate_tensors(); di, do = it.get_input_details()[0], it.get_output_details()[0]
for sid in ("e61", "u7y", "ngh"):
    qc = np.load(D / f"qc_{sid}.npy"); out = np.empty((len(qc), 256), np.int8)
    for i in range(len(qc)):
        it.set_tensor(di["index"], qc[i].reshape(1, 256, 4)); it.invoke(); out[i] = it.get_tensor(do["index"])[0]
    np.save(D / f"logit_emul_{sid}.npy", out); print(sid, len(out))
