"""Every branch of every wording table of prereg rev 7/8 is reachable (fake result dicts, no data)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from h32.confirm import conclusions


def fake(x2=False, y2=False, k1=False, po=0.5, dalt_lo=-0.1, dalt_time_lo=0.1, dropped=False, hr=False, pr=(-0.1, 0.1),
         n_k1=13, n_r=13, r_ok=True, n_p=13, p_ok=True):
    return dict(ids=["a"] * 13,
                H_K=dict(X2_reject=x2, X2_q05=0.1 if x2 else -0.1, Y2_reject=y2, Y2_q05=0.2 if y2 else -0.2, Xbar=3.0, E0bar=4.0,
                         n_cells=39, mean_rate_of_persons=1.0),
                H_K1=dict(reject=k1, q05=(0.3 if k1 else -0.3) if n_k1 else None, sum_d=1.0, n=n_k1),
                p_order=dict(p=po),
                covariates=dict(primary_dalt=dict(coef=0.5, ci90=[dalt_lo, 1.0]),
                                sens2_dalt_dtime=dict(dalt=dict(coef=0.5, ci90=[dalt_time_lo, 1.0])),
                                bootstrap_dropped=dict(dalt=dict(descriptive_only=dropped))),
                H_R=dict(R=dict(reject=hr, t_p=0.01 if hr else 0.3, mean=1.0, ci90=[0.8, 1.2], n=n_r, computable=r_ok),
                         paired=dict(ci90=list(pr), n=n_p, computable=p_ok)))


def branches(**kw):
    return {c["table"]: c["branch"] for c in conclusions(fake(**kw))}


def test_every_branch_reachable():
    seen = {}
    cases = [dict(), dict(y2=True), dict(x2=True), dict(x2=True, k1=True), dict(x2=True, po=0.01, dalt_lo=0.1),
             dict(x2=True, po=0.01, dalt_lo=0.1, dalt_time_lo=-0.1), dict(x2=True, po=0.01, dalt_lo=0.1, dropped=True),
             dict(hr=True), dict(pr=(0.1, 0.3)), dict(pr=(-0.3, -0.1)),
             dict(x2=True, n_k1=0), dict(n_r=1, r_ok=False), dict(n_p=1, p_ok=False)]
    for kw in cases:
        for t, b in branches(**kw).items(): seen.setdefault(t, set()).add(b)
    assert seen["H-K"] == {"rejected", "not rejected; Y2 rejected", "not rejected; Y2 not rejected"}
    assert seen["H-K1"] == {"not tested (fixed sequence)", "rejected", "not rejected", "not computable (n = 0)"}
    assert seen["attribution"] == {"not applied", "row 1", "row 2", "row 3"}
    assert seen["H-R"] == {"rejected", "not rejected", "not computable (n = 1)"}
    assert seen["R vs R-thin"] == {"R − R-thin > 0", "R − R-thin < 0", "CI contains 0", "not computable (n = 1)"}


def test_actual_n_in_every_at_n_sentence():
    """rev 8: every "at n = ..." is filled with the actual n of that test, never a hard-coded 13."""
    import re
    c = {x["table"]: x["sentence"] for x in conclusions(fake(x2=True, n_k1=11, n_r=12))}
    assert "at n = 11." in c["H-K1"] and "at n = 12," in c["H-R"]
    for kw in [dict(x2=True, n_k1=11, n_r=12), dict(n_r=9), dict(x2=True, n_k1=0, n_r=1, r_ok=False, n_p=1, p_ok=False)]:
        for x in conclusions(fake(**kw)):
            for m in re.findall(r"n = (\d+)", x["sentence"]):
                assert int(m) in (kw.get("n_k1", 13), kw.get("n_r", 13), kw.get("n_p", 13)) and int(m) != 13, (kw, x)
    src = (pathlib.Path(__file__).resolve().parents[1] / "h32/confirm.py").read_text()
    assert "n = 13" not in src


def test_time_clause_and_wording():
    c = {x["table"]: x for x in conclusions(fake(x2=True, po=0.01, dalt_lo=0.1, dalt_time_lo=-0.1))}
    assert "could not be separated" in c["attribution"]["sentence"]
    c = {x["table"]: x for x in conclusions(fake(y2=True))}
    assert "consistent with" not in c["H-K"]["sentence"] and "exceeded its expectation" in c["H-K"]["sentence"]
    c = {x["table"]: x for x in conclusions(fake())}
    assert "consistent with" in c["H-K"]["sentence"]
    for x in conclusions(fake(x2=True)):
        if x["table"] != "H-K1" or x["branch"] != "not tested (fixed sequence)":
            assert "same one-day route" in x["sentence"] or x["sentence"] == "(no attribution sentence)"


def test_refuses_confirmatory_while_locked():
    import subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, str(root / "05_experiments/confirm/run_confirmatory.py"), "--ids", "an0", "--label", "x"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in (r.stderr + r.stdout)
