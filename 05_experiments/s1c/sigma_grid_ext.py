"""Dev-only: extend the HMM sigma grid below its edge (eval_dev.py picked 4.0 = lower edge). Tracker only, no retraining.
    uv run python -W ignore 05_experiments/s1c/sigma_grid_ext.py"""
import sys, json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import e2cnn
R = ROOT / "05_experiments/s1c/results"; DEV = ["S10", "S11", "S12"]; SEEDS = range(5)
SIG = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz"); y = {s: Z[f"{s}_y"] for s in DEV}
m = {(s, g): float(np.mean([np.mean(np.abs(e2cnn.hmm_track_grid(np.load(R / f"fcn_proba_{s}_seed{k}.npy"), e2cnn.FCN_GRID, g) - y[s])) for k in SEEDS]))
     for s in DEV for g in SIG}
sel = {s: min(SIG, key=lambda g: np.mean([m[(o, g)] for o in DEV if o != s])) for s in DEV}
out = dict(grid=SIG, mae={f"{s}|{g}": v for (s, g), v in m.items()}, selected_loo=sel, mae_selected={s: m[(s, sel[s])] for s in DEV})
json.dump(out, open(R / "sigma_grid_ext.json", "w"), indent=2)
for s in DEV: print(s, " ".join(f"σ{g}:{m[(s,g)]:.2f}" for g in SIG), "| LOO pick", sel[s])
