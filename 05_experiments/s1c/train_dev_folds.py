"""E2-CNN LOSO on DaLiA, development folds only (test subject in S10/S11/S12), one seed per process.
    uv run python 05_experiments/s1c/train_dev_folds.py <seed>
Saves per-fold test probabilities (and the model) to 05_experiments/s1c/results/. CPU, deterministic.
"""
import sys, pathlib, json, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import e2cnn
seed = int(sys.argv[1]); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz")
subs = sorted({k.split("_")[0] for k in Z.files}, key=lambda s: int(s[1:]))
out = ROOT / "05_experiments/s1c/results"; out.mkdir(exist_ok=True, parents=True)
for test in ("S10", "S11", "S12"):
    X = np.concatenate([Z[f"{s}_X"] for s in subs if s != test]); y = np.concatenate([Z[f"{s}_y"] for s in subs if s != test])
    m = e2cnn.train_fcn(X, y, seed)
    P = e2cnn.predict_proba(m, Z[f"{test}_X"])
    np.save(out / f"fcn_proba_{test}_seed{seed}.npy", P.astype(np.float32))
    torch.save(m.state_dict(), out / f"fcn_model_{test}_seed{seed}.pt")
    print(test, seed, "argmax MAE", float(np.mean(np.abs(e2cnn.FCN_GRID[P.argmax(1)] - Z[f"{test}_y"]))), flush=True)
