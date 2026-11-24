"""T6 step 1a (project environment, read-only use of the frozen pipeline): export what the separate TF
environment needs, to 02_data/derived/t6/ (git-ignored).

- SpecFCN seed-0 weights (numpy), PLAN_T6 §1.
- DaLiA: quantisation calibration set = 500 windows, default_rng(0), from the 12 non-dev subjects;
  evaluation spectra of the dev subjects S10-S12.
- WildPPG dev e61, u7y, ngh only (confirmatory IDs are never read): resampled streams (PPG 64 Hz, ACC 32 Hz)
  exactly as confirm.load_recording, their spectra, ACC states.
- Float PyTorch logits (194 masked bins) for every evaluation set.
    uv run python -W ignore 05_experiments/t6_mcu/export_inputs.py
"""
import pathlib, sys
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import e2cnn, features, io_dalia, confirm, method as M  # noqa: E402

OUT = ROOT / "02_data/derived/t6"; OUT.mkdir(parents=True, exist_ok=True)
DEV_W = ("e61", "u7y", "ngh")
torch.set_num_threads(8)

m = e2cnn.SpecFCN(); m.load_state_dict(torch.load(ROOT / "05_experiments/s1c/results/fcn_all_seed0.pt")); m.eval()
np.savez(OUT / "weights_seed0.npz", **{k: v.numpy() for k, v in m.state_dict().items()})


def logits(X):
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(X[i:i + 4096])).numpy() for i in range(0, len(X), 4096)])


Z = np.load(ROOT / "02_data/derived/dalia_spectra.npz")
non_dev = [s for s in io_dalia.CONFIRMATORY]          # S1-S9, S13-S15 (DaLiA is train/dev only since v2.1)
assert set(non_dev).isdisjoint(io_dalia.DEV)
pool = np.concatenate([Z[f"{s}_X"] for s in non_dev])
idx = np.sort(np.random.default_rng(0).choice(len(pool), 500, replace=False))
np.save(OUT / "calib_dalia_500.npy", pool[idx])
for s in io_dalia.DEV:
    X = Z[f"{s}_X"]; np.savez(OUT / f"eval_dalia_{s}.npz", X=X, y=Z[f"{s}_y"], logit_float=logits(X))
    print(s, X.shape)

for sid in DEV_W:
    assert sid in confirm.W.DEV
    ppg, acc, n, _ = confirm.load_recording(sid)
    X = features.spectra(ppg, acc, n); st = M.states(acc, n)[:len(X)]
    np.savez(OUT / f"eval_wild_{sid}.npz", X=X, st=st, logit_float=logits(X),
             ppg64=ppg.astype(np.float32), acc32=acc.astype(np.float32), n=len(X))
    print(sid, X.shape)
print("calib", pool[idx].shape, "from", len(pool))
