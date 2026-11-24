"""Unit tests on synthetic data only (no dataset access)."""
import sys, pathlib, pickle, io
import numpy as np, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from h32 import events, stats, e1, io_dalia

def test_episodes_merge_and_kmin():
    f = np.zeros(40, bool); f[2:4] = True; f[6:9] = True; f[30] = True
    assert events.episodes(f, k_min=3, merge_gap=2) == [(2, 8)]
    assert events.episodes(f, k_min=1, merge_gap=0) == [(2, 3), (6, 8), (30, 30)]

def test_match():
    ref = [(10, 15)]; al = [(14, 16), (40, 42)]
    assert events.match(al, ref, tol_s=0) == (1, 1)

def test_conformal_quantile():
    s = np.arange(1, 100)  # n=99
    assert stats.conformal_quantile(s, 0.1) == 90.0
    assert stats.conformal_quantile(np.arange(5), 0.01) == float("inf")

def test_poisson_ci_zero():
    r, lo, hi = stats.poisson_rate_ci(0, 10.0)
    assert r == 0 and lo == 0 and abs(hi - 0.3689) < 1e-3

def test_e1_recovers_sinusoid():
    fs, T = 64, 120; t = np.arange(T * fs) / fs
    bvp = np.sin(2 * np.pi * 1.5 * t)                 # 90 bpm
    acc = np.tile([0, 0, 64.0], (T * 32, 1))          # at rest, 1/64 g units
    n = (T - 8) // 2 + 1
    hr, mot = e1.estimate(bvp, acc, n)
    assert np.nanmax(np.abs(hr - 90)) < 1.0 and not mot.any()

def test_unpickler_refuses_arbitrary_globals():
    payload = pickle.dumps(io.StringIO("x"))
    with pytest.raises(pickle.UnpicklingError):
        io_dalia._SafeUnpickler(io.BytesIO(payload)).load()

def test_e2_recovers_sinusoid_with_harmonic():
    from h32 import e2
    fs, T = 64, 120; t = np.arange(T * fs) / fs
    bvp = np.sin(2 * np.pi * 1.5 * t) + 0.6 * np.sin(2 * np.pi * 3.0 * t)   # 90 bpm + 2nd harmonic
    acc = np.tile([0, 0, 1.0], (T * 32, 1))
    n = (T - 8) // 2 + 1
    hr = e2.estimate(bvp, acc, n)
    assert np.nanmax(np.abs(hr[5:] - 90)) <= 1.0

def test_inject_raises_pulse_rate():
    from h32 import inject, e1
    fs, T = 64, 600; t = np.arange(T * fs) / fs
    ppg = np.sin(2 * np.pi * 1.25 * t)                    # 75 bpm
    nwin = (T - 8) // 2 + 1
    ref = np.full(nwin, 75.0)
    y, ref2, (a, b) = inject.inject(ppg, fs, ref, t0=200.0, delta=30.0, ramp_s=30, plateau_s=120)
    acc = np.tile([0, 0, 1.0], (T * 32, 1))
    hr, _ = e1.estimate(y, acc, nwin)
    plateau = [(i * 2 + 4 > 235) and (i * 2 + 4 < 345) for i in range(nwin)]
    assert abs(np.median(hr[plateau]) - 105) <= 2
    assert abs(np.median(hr[: 90]) - 75) <= 1 and abs(np.median(hr[-60:]) - 75) <= 1   # alignment restored
    assert abs(np.max(ref2) - 105) < 1e-6


def test_method_lemma1_one_alarm_per_block_and_exceedance():
    """Lemma 1 (03_theory/R1.md): <= 1 alarm start per state-time block; alarm => block max > tau."""
    import numpy as np
    from h32 import method as M
    rng = np.random.default_rng(0); n = 20000
    st = np.repeat(rng.integers(0, 3, n // 50), 50)[:n]          # state runs of 100 s -> fragmented blocks
    mon = rng.random(n) > 0.1
    hp = 80 + 10 * rng.standard_normal(n)
    blk = M.blocks(mon, st, lw=100)
    D = M.debounce(hp, mon, st, blk)
    S = M.block_scores(D, st, blk)
    nc = M.complete_blocks(mon, st, blk, lw=100)
    tau = M.main_thresholds(S, nc, n_cal=6, kstar=6)
    tm = M.test_mask(st, blk, tau, nc)
    a = M.alarms(D, st, blk, M.thr_from_state(st, tau), tm)
    keys = [(st[i], blk[i]) for i in a]
    assert len(keys) == len(set(keys))
    for i in a:
        assert S[st[i]][blk[i]] > tau[st[i]]
    exceed = sum(int(np.sum(S[k][6:nc[k]] > tau[k])) for k in range(3) if tau[k] is not None)
    assert abs(M.monitored_hours(mon, tm) - sum(max(nc[k] - 6, 0) for k in range(3) if tau[k] is not None) * 100 * 2 / 3600) < 1e-9
    assert len(a) <= exceed
