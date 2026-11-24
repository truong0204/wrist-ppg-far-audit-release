"""prereg rev 8 §4.0 (PI 23-11): persons and cells with missing data, and unreadable raw files.

Synthetic persons are built from the dev dry-run arrays (02_data/derived/confirm_cache/dryrun_dev_1; skipped if absent):
- x_none : readable, but no monitored window in any of the 5 seeds (SQI set to 0)   -> no block in all 5 seeds
- x_two  : u7y with SQI set to 0 in seeds 2, 3, 4                                     -> blocks in 2 of 5 seeds
- x_one  : e61 with only the windows of rest blocks 0..6 monitored                    -> exactly 1 K test block
"""
import sys, pathlib
import numpy as np, pytest, scipy.io as sio
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from h32 import confirm as C, io_wildppg as W

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "02_data/derived/confirm_cache/dryrun_dev_1"
DEV = ["e61", "u7y", "ngh"]
SQ = ["sq", "sq20", "sq30", "sq40", "sq_cal"]


def _mask(z, keep):
    z = dict(z)
    for key in SQ: z[key] = np.where(keep[:len(z[key])], z[key], 0.0).astype(z[key].dtype)
    return z


@pytest.fixture(scope="module")
def run():
    if not all((CACHE / f"{s}_seed{k}.npz").exists() for s in DEV for k in C.SEEDS):
        pytest.skip("dev dry-run cache not present")
    arr = {(s, k): dict(np.load(CACHE / f"{s}_seed{k}.npz")) for s in DEV for k in C.SEEDS}
    for k in C.SEEDS:
        z = arr[("ngh", k)]; arr[("x_none", k)] = _mask(z, np.zeros(len(z["st"]), bool))
        z = arr[("u7y", k)]; arr[("x_two", k)] = z if k in (0, 1) else _mask(z, np.zeros(len(z["st"]), bool))
        z = arr[("e61", k)]; cl = C.stream(z["st"], z["sq"], z["hp3"])
        arr[("x_one", k)] = _mask(z, cl["mon"] & (z["st"] == 0) & (cl["blk"] >= 0) & (cl["blk"] <= 6))
    cov = {s: C.load_covariates(s) for s in DEV}; cov.update(x_none=cov["ngh"], x_two=cov["u7y"], x_one=cov["e61"])
    ids = DEV + ["x_none", "x_two", "x_one"]
    D = C.Data(ids, arrays=arr, cov=cov)
    return D, C.analyse(ids, log=lambda m: None, D=D, excluded={"x_bad": "attempt 1: KeyError; attempt 2: KeyError"})


def test_construction(run):
    D, _ = run
    assert all(len(S) == 0 for k in C.SEEDS for S in D.S[("x_none", k)])
    assert [sum(len(S) for S in D.S[("x_two", k)]) > 0 for k in C.SEEDS] == [True, True, False, False, False]
    assert all([len(S) for S in D.S[("x_one", k)]] == [7, 0, 0] for k in C.SEEDS)


def test_hk_keeps_everyone_sum_type(run):
    D, res = run; hk = res["H_K"]
    assert hk["n"] == 6 and hk["d_i"]["x_none"] == 0.0
    cells = C.cells_rule(D, "K")                                             # sum-type: seeds 2-4 contribute 0, divided by 5
    for c in cells: c["P0"] = C.p0_exceed(c["T"])
    per = [sum((c["rate"] > 2) - c["P0"] for c in cells if c["sid"] == "x_two" and c["seed"] == k) for k in C.SEEDS]
    assert per[2:] == [0, 0, 0] and hk["d_i"]["x_two"] == pytest.approx(sum(per) / 5)


def test_hk1_excludes_only_persons_without_blocks_in_all_seeds(run):
    _, res = run; k1 = res["H_K1"]
    assert set(k1["excluded_no_blocks"]) == {"x_none", "x_one"} and k1["n"] == 4 and "x_two" in k1["d_i"]


def test_rate_type_over_seeds_with_blocks(run):
    D, res = run; hr = res["H_R"]
    for rule in ("R", "Rthin"):
        assert set(hr[rule]["excluded_no_blocks"]) == {"x_none", "x_one"} and hr[rule]["n"] == 4
        assert hr[rule]["n_seeds_per_person"]["x_two"] == 2
        cells = C.cells_rule(D, rule)
        N = sum(c["N"] for c in cells if c["sid"] == "u7y" and c["seed"] in (0, 1)); T = sum(c["T"] for c in cells if c["sid"] == "u7y" and c["seed"] in (0, 1))
        assert hr[rule]["per_person"]["x_two"] == pytest.approx(6 * N / T)           # not pulled down by 0 for seeds 2-4
        assert hr[rule]["pooled_rate"] is not None and "x_two" in hr[rule]["test_blocks_per_person"]
    p = hr["paired"]
    assert set(p["excluded_no_blocks"]) == {"x_none", "x_one"} and p["n"] == 4 and p["per_person"]["x_two"]["n_seeds"] == 2


def test_one_test_block_person(run):
    _, res = run; far = res["FAR"]["K"]
    assert far["test_blocks_per_person"]["x_one"] == 1.0 and far["n_seeds_per_person"]["x_one"] == 5
    assert far["n_seeds_per_person"]["x_none"] == 0 and far["per_person"]["x_none"] != far["per_person"]["x_none"]   # NaN, not 0
    assert far["n_persons"] == 5


def test_descriptive_tables_report_actual_n(run):
    _, res = run
    md = res["sensitivity"]["matched_diff"]
    assert all(v["n"] <= 5 and "x_none" not in v["per_person"] for v in md.values())
    assert all(r["sid"] != "x_none" for r in res["ACI_bound"]["cells"])
    assert all(n <= 5 for n in res["covariates"]["n_persons_per_seed"].values())
    assert 0 < res["p_order"]["p"] <= 1


def test_actual_n_in_sentences_and_deviation(run):
    _, res = run
    c = {x["table"]: x["sentence"] for x in res["conclusions"]}
    br = {x["table"]: x["branch"] for x in res["conclusions"]}
    assert br["H-R"] == "not rejected" and "at n = 4," in c["H-R"]
    assert res["excluded_unreadable"] == {"x_bad": "attempt 1: KeyError; attempt 2: KeyError"} and "x_bad" in res["deviations"][0]
    assert "x_bad" not in res["ids"]


@pytest.mark.parametrize("content", ["no_wrist", "no_ppg", "garbage"])
def test_unreadable_file_is_retried_then_excluded(tmp_path, monkeypatch, content):
    raw = tmp_path / "raw"; raw.mkdir(); f = raw / "WildPPG_Part_e61.mat"
    if content == "no_wrist": sio.savemat(f, {"other": np.arange(3)})
    elif content == "no_ppg": sio.savemat(f, {"wrist": {"acc_x": {"v": np.arange(3.0), "fs": 32, "descr": "x"}}})
    else: f.write_bytes(b"not a mat file at all")
    monkeypatch.setattr(W, "RAW", raw); logs = []; cache = tmp_path / "cache"
    r = C.infer("e61", cache, log=logs.append)
    assert isinstance(r, str) and "attempt 1" in r and "attempt 2" in r
    assert any("UNREADABLE e61" in m for m in logs) and not list(cache.glob("*.npz*"))


def test_lock_is_never_caught_as_unreadable(tmp_path):
    assert not W.CONFIRMATORY_PPG_OPEN
    with pytest.raises(PermissionError):
        C.infer("an0", tmp_path / "cache", log=lambda m: None)
