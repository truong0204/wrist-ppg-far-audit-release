"""Build the DaLiA spectra cache used by all S1c scripts (all 15 subjects; v2.1 makes DaLiA train/dev only).
    uv run python 05_experiments/s1c/build_spectra.py [out.npz]
Default output: 02_data/derived/dalia_spectra.npz (git-ignored, derived). Keys: S<i>_X [n,4,256], S<i>_y [n].
Same code as the inline cache build of 21-11 (S1c); written down here so `make s1c` can rebuild it.
"""
import sys, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_dalia, features
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "02_data/derived/dalia_spectra.npz"
F = {}
for s in sorted(io_dalia.DEV + io_dalia.CONFIRMATORY, key=lambda s: int(s[1:])):
    d = io_dalia.load_subject(s); n = len(d["label"])
    X = features.spectra(d["bvp"], d["acc"], n); F[s] = (X, d["label"][:len(X)])
    print(s, X.shape, flush=True)
np.savez_compressed(out, **{f"{s}_X": v[0] for s, v in F.items()}, **{f"{s}_y": v[1] for s, v in F.items()})
print("wrote", out)
