"""Final E2 (SpecFCN, amendment 1) trained on all 15 DaLiA subjects, one seed per process (prereg v2.1 §3:
"The final E2 weights for WildPPG are trained on all DaLiA subjects"). No WildPPG data is read here.
    uv run python -W ignore 05_experiments/s1c/train_all_dalia.py <seed>"""
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import e2cnn
seed = int(sys.argv[1]); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz")
subs = sorted({k.split("_")[0] for k in Z.files}, key=lambda s: int(s[1:]))
X = np.concatenate([Z[f"{s}_X"] for s in subs]); y = np.concatenate([Z[f"{s}_y"] for s in subs])
m = e2cnn.train_fcn(X, y, seed)
out = ROOT / "05_experiments/s1c/results"; torch.save(m.state_dict(), out / f"fcn_all_seed{seed}.pt")
print("all15", seed, "n_train", len(y), "train argmax MAE", float(np.mean(np.abs(e2cnn.FCN_GRID[e2cnn.predict_proba(m, X).argmax(1)] - y))), flush=True)
