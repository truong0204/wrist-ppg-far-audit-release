"""Single entry point of the confirmatory analysis (prereg v2.2, rev 8). The dev dry run and the confirmatory run go
through the SAME code path (04_code/h32/confirm.py); the only parameter is the list of WildPPG IDs.

    uv run python 05_experiments/confirm/run_confirmatory.py --ids e61 u7y ngh --label dryrun_dev_1
    uv run python 05_experiments/confirm/run_confirmatory.py --ids <13 confirmatory IDs> --label confirmatory

Refuses any confirmatory ID while io_wildppg.CONFIRMATORY_PPG_OPEN is False (checked before any file is read).
Outputs (05_experiments/confirm/runs/<label>/): results.json (deterministic), conclusions.md, run.log, meta.json
(timestamps, versions; the only file allowed to differ between two identical runs).
"""
import sys, json, time, argparse, pathlib, platform, subprocess, hashlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "04_code"))
from h32 import io_wildppg as W

ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--label", required=True)
a = ap.parse_args()
bad = [i for i in a.ids if i not in W.DEV + W.CONFIRMATORY]
if bad: sys.exit(f"unknown IDs: {bad}")
locked = [i for i in a.ids if i in W.CONFIRMATORY and not W.CONFIRMATORY_PPG_OPEN]
if locked: sys.exit(f"REFUSED: confirmatory IDs {locked} requested while CONFIRMATORY_PPG_OPEN = False (prereg v2.2)")

from h32 import confirm as C
out = ROOT / "05_experiments/confirm/runs" / a.label; out.mkdir(parents=True, exist_ok=True)
cache = ROOT / "02_data/derived/confirm_cache" / a.label
logf = open(out / "run.log", "a")   # rev 8: a technical rerun appends; the cause is added to this log with its time
def log(msg):
    print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

t0 = time.time(); log(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}"); log(f"IDs: {a.ids}")
done0 = sorted(f.name for f in cache.glob("*.npz")) if cache.exists() else []
if done0: log(f"rerun: {len(done0)} completed arrays reused: {done0}")
excluded = {}                                     # rev 8: unreadable raw file -> excluded from every analysis
for sid in a.ids:
    r = C.infer(sid, cache, log)
    if isinstance(r, str): excluded[sid] = r
ids_ok = [s for s in a.ids if s not in excluded]
t_inf = time.time() - t0; log("inference done" + (f"; excluded (unreadable): {sorted(excluded)}" if excluded else ""))
res = C.analyse(ids_ok, cache, log, excluded=excluded); res["ids_requested"] = list(a.ids); t_all = time.time() - t0


def clean(o):
    import numpy as np
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): o = float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, float) and o != o: return None
    return o


js = json.dumps(clean(res), indent=1, sort_keys=True, allow_nan=False)
(out / "results.json").write_text(js)
md = ["# Conclusions (filled automatically)", "", f"IDs: {', '.join(ids_ok)}", ""]
md += [f"Excluded (unreadable raw file, deviation): {', '.join(sorted(excluded))}", ""] if excluded else []
for c in res["conclusions"]:
    md += [f"## {c['table']} — branch: {c['branch']}", f"Reason: {c['reason']}", "", c["sentence"], ""]
(out / "conclusions.md").write_text("\n".join(md))
for c in res["conclusions"]:
    log(f"[{c['table']}] branch = {c['branch']} | because {c['reason']}")
script_files = [ROOT / "04_code/h32/confirm.py", pathlib.Path(__file__).resolve(), ROOT / "04_code/h32/method.py"]
meta = dict(label=a.label, ids=a.ids, started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            seconds_inference=round(t_inf, 1), seconds_total=round(t_all, 1), python=platform.python_version(),
            platform=platform.platform(), git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in script_files},
            results_sha256=hashlib.sha256(js.encode()).hexdigest())
(out / "meta.json").write_text(json.dumps(meta, indent=1))
log(f"done in {t_all:.0f} s (inference {t_inf:.0f} s); results sha256 {meta['results_sha256']}")
