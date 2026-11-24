"""H32 S3a: every number of the manuscript comes from here (no number is typed into main.tex).

Reads only:
  05_experiments/confirm/runs/confirmatory/results.json   (confirmatory run, SHA256 checked below)
  05_experiments/confirm/runs/confirmatory/conclusions.md  (fixed sentences, verbatim)
  05_experiments/DEVIATIONS.md                            (sentence D2 and the H-R sentence F2, verbatim)
  05_experiments/t6_mcu/results/ and t6_mcu/model/       (section 5, T6 microcontroller: development participants,
                                                          seed 0; percentiles of stored board timings, sums, ratios)
Writes:
  07_manuscript/generated/numbers.tex    \newcommand per number
  07_manuscript/generated/sentences.tex  fixed sentences, verbatim (LaTeX-escaped only)
  07_manuscript/NUMBERS.md               macro | value | JSON key (or other source), for checking

Nothing is computed from the 13 persons' data; the only operations on results.json values are rounding and, where
stated in the key column, min/max over values that are themselves in results.json.
    uv run python 07_manuscript/make_numbers.py
"""
import hashlib, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "05_experiments/confirm/runs/confirmatory"
OUT = ROOT / "07_manuscript/generated"; OUT.mkdir(exist_ok=True)
SHA = "2460459493a626f8e6205a1fbdc1080278378e990ec8c5d5ac013cfee77cffb2"
raw = (RUN / "results.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == SHA, "results.json is not the confirmatory file"
R = json.loads(raw)
IDS = R["ids"]


def get(path):
    v = R
    for p in path.split("/"):
        v = v[int(p)] if isinstance(v, list) else v[p]
    return v


ROWS = []   # (macro, text, source)


DIG = dict(zip("0123456789", ["zero","one","two","three","four","five","six","seven","eight","nine"]))


def clean(m):
    m = "".join(DIG.get(ch, ch) for ch in m)
    assert m.isalpha(), m
    return m


def num(macro, path, fmt="{:.2f}", src=None, value=None):
    v = get(path) if value is None else value
    s = fmt.format(v) if isinstance(v, (int, float)) else str(v)
    if s.startswith("-") and float(s) == 0:                  # no "-0.00" from float round-off
        s = s[1:]
    s = s.replace("-", "−")
    ROWS.append((clean(macro), s, src or f"`{path}`"))
    return s


def mm(macro, paths, fmt="{:.2f}", what="min–max"):
    vals = [get(p) for p in paths]
    s = f"{fmt.format(min(vals))}–{fmt.format(max(vals))}"
    common = paths[0].rsplit("/", 2)
    ROWS.append((clean(macro), s, f"{what} over {len(vals)} values `{paths[0]}` … `{paths[-1]}`"))
    return s


def lit(macro, s, src):
    ROWS.append((clean(macro), s, src))

# ---------------------------------------------------------------- H-K (X2, X1, Y2)
num("nXbar", "H_K/Xbar"); num("nEzero", "H_K/E0bar"); num("nCells", "H_K/n_cells", "{:d}")
num("nPersons", "H_K/n", "{:d}"); num("nXtwoQ", "H_K/X2_q05"); num("nXtwoSum", "H_K/sum_d", "{:.2f}")
num("nXoneMedP", "H_K/X1_median_p", "{:.3f}")
for i, row in enumerate(get("H_K/X1")):
    tag = "ABCDE"[i]
    num(f"nXoneX{tag}", f"H_K/X1/{i}/X", "{:d}"); num(f"nXoneP{tag}", f"H_K/X1/{i}/p", "{:.3f}")
    num(f"nXoneE{tag}", f"H_K/X1/{i}/E0", "{:.2f}")
num("nYtwoSum", "H_K/Y2_sum"); num("nYtwoQ", "H_K/Y2_q05")
num("nKmean", "FAR/K/mean_of_persons"); num("nKlo", "FAR/K/cluster_ci95/0"); num("nKhi", "FAR/K/cluster_ci95/1")
num("nKpooled", "FAR/K/pooled")
# ---------------------------------------------------------------- H-K1
num("nHKoneSum", "H_K1/sum_d"); num("nHKoneQ", "H_K1/q05"); num("nHKoneN", "H_K1/n", "{:d}")
num("nHKoneQmNine", "H_K1/d_i/qm9")
num("nKmOneMean", "FAR/Km1/mean_of_persons"); num("nKmOneLo", "FAR/Km1/cluster_ci95/0"); num("nKmOneHi", "FAR/Km1/cluster_ci95/1")
n_km1_cells = len(get("FAR/Km1/per_cell"))
num("nKmOneCells", "", "{:d}", src="number of keys of `FAR/Km1/per_cell`", value=n_km1_cells)
short = sum(1 for c in get("FAR/Km1/per_cell").values() if c["T"] <= 3)
num("nKmOneShort", "", "{:d}", src="number of cells in `FAR/Km1/per_cell` with `T` ≤ 3", value=short)
# ---------------------------------------------------------------- attribution
num("nPorder", "p_order/p", "{:.3f}"); num("nXbarOracle", "p_order/Xbar_oracle_mean")
num("nDaltCoef", "covariates/primary_dalt/coef"); num("nDaltLo", "covariates/primary_dalt/ci90/0"); num("nDaltHi", "covariates/primary_dalt/ci90/1")
num("nSoneDalt", "covariates/sens1_dalt_dtemp/dalt/coef"); num("nSoneDaltLo", "covariates/sens1_dalt_dtemp/dalt/ci90/0")
num("nSoneDaltHi", "covariates/sens1_dalt_dtemp/dalt/ci90/1")
num("nSoneDtemp", "covariates/sens1_dalt_dtemp/dtemp/coef", "{:.3f}"); num("nSoneDtempLo", "covariates/sens1_dalt_dtemp/dtemp/ci90/0", "{:.3f}")
num("nSoneDtempHi", "covariates/sens1_dalt_dtemp/dtemp/ci90/1", "{:.3f}")
for m, key in (("Dalt", "dalt"), ("DaltDtemp", "dalt+dtemp"), ("DaltDtime", "dalt+dtime"), ("Dtemp", "dtemp"), ("Dtime", "dtime")):
    num(f"nDrop{m}", f"covariates/bootstrap_dropped/{key}/dropped", "{:d}")
    num(f"nDropPct{m}", f"covariates/bootstrap_dropped/{key}/frac", "{:.1%}".replace("%", "") if False else "{:.4g}")
pct = lambda p: num(p[0], p[1], "{:.1f}", src=f"100 × `{p[1]}`", value=100 * get(p[1]))
pct(("nDropPctDaltDtimeP", "covariates/bootstrap_dropped/dalt+dtime/frac")); pct(("nDropPctDtimeP", "covariates/bootstrap_dropped/dtime/frac"))
for i, t in enumerate(("Low", "Mid", "High")):
    num(f"nTerc{t}", f"covariates/exceed_by_dalt_tercile/{i}")
num("nTercEdgeA", "covariates/tercile_edges_km/0"); num("nTercEdgeB", "covariates/tercile_edges_km/1")
num("nCorrDaltDtime", "covariates/corr_dalt_dtime"); num("nCorrDaltDtemp", "covariates/corr_dalt_dtemp")
num("nCovBlocks", "covariates/n_test_blocks_per_seed", "{:d}")
num("nAssocRho", "descriptive_drift/association/rho/spearman"); num("nAssocRhoLo", "descriptive_drift/association/rho/ci95/0")
num("nAssocRhoHi", "descriptive_drift/association/rho/ci95/1")
# ---------------------------------------------------------------- H-R, R-thin, paired, ACF
num("nRmean", "H_R/R/mean"); num("nRlo", "H_R/R/ci90/0"); num("nRhi", "H_R/R/ci90/1"); num("nRp", "H_R/R/t_p", "{:.3f}")
num("nRwil", "H_R/R/wilcoxon_p", "{:.3f}"); num("nRboot", "H_R/R/boot_percentile_q05", "{:.3f}"); num("nRpooled", "H_R/R/pooled_rate", "{:.3f}")
num("nRthinMean", "H_R/Rthin/mean", "{:.3f}"); num("nRthinLo", "H_R/Rthin/ci90/0", "{:.3f}"); num("nRthinHi", "H_R/Rthin/ci90/1", "{:.3f}")
num("nRthinP", "H_R/Rthin/t_p", "{:.3f}")
num("nPairDiff", "H_R/paired/diff_mean", "{:.3f}"); num("nPairLo", "H_R/paired/ci90/0", "{:.3f}"); num("nPairHi", "H_R/paired/ci90/1", "{:.3f}")
for kind, t in (("raw", "Raw"), ("detrended", "Det")):
    num(f"nAcf{t}", f"H_R/acf/{kind}/pooled/lag1/r"); num(f"nAcf{t}Lo", f"H_R/acf/{kind}/pooled/lag1/ci90/0")
    num(f"nAcf{t}Hi", f"H_R/acf/{kind}/pooled/lag1/ci90/1")
# ---------------------------------------------------------------- FAR of the other rules
for rule, t in (("R", "R"), ("Rthin", "Rthin"), ("Rsel", "Rsel"), ("ACI", "ACI")):
    num(f"nFar{t}", f"FAR/{rule}/mean_of_persons"); num(f"nFar{t}Lo", f"FAR/{rule}/cluster_ci95/0"); num(f"nFar{t}Hi", f"FAR/{rule}/cluster_ci95/1")
num("nAciVacuous", "ACI_bound/n_vacuous", "{:d}")
# ---------------------------------------------------------------- sensitivity, matched FAR, masking, contamination
for d in ("20", "30", "40"):
    t = {"20": "A", "30": "B", "40": "C"}[d]
    num(f"nEvents{t}", f"sensitivity/summary/{d}|all/n_events", "{:d}")
    for rule in ("K", "R", "B0", "Rthin", "Rsel", "ACI", "B0sm", "R_self", "K_self"):
        num(f"nSens{rule.replace('_', '')}{t}", f"sensitivity/summary/{d}|all/sens_{rule}", "{:.3f}")
    for cmp_, c in (("K-B0m", "Bm"), ("K-B0sm", "Bsm")):
        md = get(f"sensitivity/matched_diff/{d}|{cmp_}")
num_md = get("sensitivity/matched_diff/20|K-B0sm")
for d in ("20", "30", "40"):
    t = {"20": "A", "30": "B", "40": "C"}[d]
    for cmp_, c in (("K-B0m", "Bm"), ("K-B0sm", "Bsm")):
        md = get(f"sensitivity/matched_diff/{d}|{cmp_}")
        keys = list(md)
        mk = "mean" if "mean" in md else keys[0]
        num(f"nMd{c}{t}", f"sensitivity/matched_diff/{d}|{cmp_}/{mk}", "{:.3f}")
        ck = "ci95" if "ci95" in md else ("cluster_ci95" if "cluster_ci95" in md else None)
        if ck:
            num(f"nMd{c}{t}Lo", f"sensitivity/matched_diff/{d}|{cmp_}/{ck}/0", "{:.3f}")
            num(f"nMd{c}{t}Hi", f"sensitivity/matched_diff/{d}|{cmp_}/{ck}/1", "{:.3f}")
hs = [str(h) for h in range(2, 8)]
mm("nMaskAfterR", [f"masking/R/{h}/after" for h in hs], "{:.3f}")
mm("nMaskNoPriorR", [f"masking/R/{h}/no_prior" for h in hs], "{:.3f}")
for rule in ("K", "R", "Rthin", "Rsel", "ACI"):
    num(f"nMaskN{rule}", f"masking/{rule}/2/n", "{:d}")
for d, t in (("20", "A"), ("30", "B"), ("40", "C")):
    num(f"nContClean{t}", f"contamination/sens/{d}/clean_cal", "{:.3f}"); num(f"nContCont{t}", f"contamination/sens/{d}/contaminated_cal", "{:.3f}")
dt = get("contamination/d_tau")
num("nDtauN", "", "{:d}", src="length of `contamination/d_tau`", value=len(dt))
num("nDtauZero", "", "{:d}", src="number of zeros in `contamination/d_tau`", value=sum(1 for x in dt if x == 0))
# ---------------------------------------------------------------- coverage of wear time
for rule, t in (("K", "K"), ("R", "R"), ("Rthin", "Rthin")):
    mm(f"nFrac{t}", [f"coverage/per_person_frac_wear/{s}/{rule}" for s in IDS])
cells = get("coverage/cells")
for st, t in (("rest", "Rest"), ("light", "Light"), ("vigorous", "Vig")):
    idx = [i for i, c in enumerate(cells) if c["state"] == st]
    mm(f"nCalK{t}", [f"coverage/cells/{i}/K_cal_done_h" for i in idx], "{:.1f}")
num("nRmeanThree", "H_R/R/mean", "{:.3f}"); num("nRloThree", "H_R/R/ci90/0", "{:.3f}"); num("nRhiThree", "H_R/R/ci90/1", "{:.3f}")
mm("nCalKAll", [f"coverage/cells/{i}/K_cal_done_h" for i, c in enumerate(cells) if c["K_cal_done_h"] is not None], "{:.1f}")
num("nCalKCells", "", "{:d}", src="number of entries of `coverage/cells` with `K_cal_done_h` not null",
    value=sum(1 for c in cells if c["K_cal_done_h"] is not None))
# ---------------------------------------------------------------- numbers that come from the frozen plan, not from results
lit("nTheta", "0.0290", "prereg v2.2 §1 (θ = 0.02896, rounded as in §1 and NOTES §5)")
lit("nGuar", "0.912", "prereg v2.2 §2 (K guarantee, R1.md Prop. 2a with n′ = 6, k* = 6)")
lit("nRbound", "0.857", "prereg v2.2 §2 ((60/10)/7)")
lit("nXtwoSizeA", "0.096", "prereg v2.2 §4.1b table, P(H-K rejects), MA(1) lag-1 = 0.17")
lit("nXtwoSizeB", "0.166", "prereg v2.2 §4.1b table, P(H-K rejects), MA(1) lag-1 = 0.30")
lit("nPowerHK", "0.75", "prereg v2.2 §4.6 table (f = 0.2, 1.5 r, X2)")
lit("nMaOneA", "0.172", "03_theory/R1.md §4b table, MA(1) θ = 0.5, P(exceed) R window 0.1718")
lit("nMaOneB", "0.182", "03_theory/R1.md §4b table, MA(1) θ = 1.0, P(exceed) R window 0.1821")

# ---------------------------------------------------------------- fixed sentences, verbatim
TEXREP = [("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"),
          ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("−", r"$-$"), ("τ", r"$\tau$"), ("§", r"\S"), ("Δ", r"$\Delta$"),
          ("·", r"$\cdot$"), ("×", r"$\times$")]


def tex(s):
    for a, b in TEXREP:
        s = s.replace(a, b)
    s = re.sub(r'(^|(?<=[\s(\[]))"', '``', s); s = s.replace('"', "''")
    return s


conc = (RUN / "conclusions.md").read_text()
blocks = re.findall(r"## (.+?) — branch: (.+?)\nReason: (.+?)\n\n(.+?)(?=\n\n## |\Z)", conc, re.S)
names = {"H-K": "HK", "H-K1": "HKone", "attribution": "Attr", "H-R": "HR", "R vs R-thin": "RRthin"}
L1 = ("All participants of this dataset followed the same one-day route with an altitude change of about 3,000 m, so "
      "the persons are not independent in the direction of any change over the day, and the result may not carry over "
      "to free-living wear without such a change.")
S = ["% generated by 07_manuscript/make_numbers.py from conclusions.md and DEVIATIONS.md (verbatim; LaTeX escaping only)"]
for title, branch, reason, body in blocks:
    body = body.strip(); assert body.endswith(L1), title
    k = names[title]
    S.append(f"\\newcommand{{\\conc{k}}}{{{tex(body)}}}")
    S.append(f"\\newcommand{{\\conc{k}Branch}}{{{tex(branch)}}}")
    S.append(f"\\newcommand{{\\conc{k}Reason}}{{{tex(reason)}}}")
    S.append(f"\\newcommand{{\\conc{k}NoLone}}{{{tex(body[:-len(L1)].strip())}}}")
assert len(blocks) == 5
S.append(f"\\newcommand{{\\sentLone}}{{{tex(L1)}}}")
dev = (ROOT / "05_experiments/DEVIATIONS.md").read_text()
d2 = re.search(r"Sentence added to the paper:\n\n((?:> .*\n)+)", dev).group(1)
f2 = re.search(r"\*\*Sentence used in the paper \(F2.*?\n\n((?:> .*\n)+)", dev).group(1)
unq = lambda b: " ".join(l[2:].strip() for l in b.strip().splitlines()).strip().strip('"').strip()
D2, F2 = unq(d2), unq(f2)
S.append(f"\\newcommand{{\\sentDtwo}}{{{tex(D2)}}}")
S.append(f"\\newcommand{{\\sentFtwo}}{{{tex(F2)}}}")
(OUT / "sentences.tex").write_text("\n".join(S) + "\n")

# numbers inside the fixed sentences: each must be the rounded value of its key (checked here)
CHK = [("H-K sentence", "8.80", "H_K/Xbar", "{:.2f}"), ("H-K sentence", "39", "H_K/n_cells", "{:d}"),
       ("H-K sentence", "4.39", "H_K/E0bar", "{:.2f}"), ("H-K reason", "1.03", "H_K/X2_q05", "{:.2f}"),
       ("H-K1 reason", "6.54", "H_K1/q05", "{:.2f}"), ("attribution reason", "0.004", "p_order/p", "{:.3f}"),
       ("attribution reason", "0.14", "covariates/primary_dalt/ci90/0", "{:.2f}"),
       ("attribution reason", "0.88", "covariates/primary_dalt/ci90/1", "{:.2f}"),
       ("H-R sentence / F2", "0.88", "H_R/R/mean", "{:.2f}"), ("H-R sentence / F2", "0.74", "H_R/R/ci90/0", "{:.2f}"),
       ("H-R sentence / F2", "1.03", "H_R/R/ci90/1", "{:.2f}"), ("H-R reason", "0.383", "H_R/R/t_p", "{:.3f}"),
       ("R vs R-thin reason", "-0.22", "H_R/paired/ci90/0", "{:.2f}"), ("R vs R-thin reason", "0.0", "H_R/paired/ci90/1", "{:.1f}"),
       ("D2", "14.4", "covariates/bootstrap_dropped/dalt+dtime/frac", "x100:{:.1f}")]
for d in ("20", "30", "40"):
    for rule in ("R", "K"):
        CHK.append(("F2", f"{get(f'sensitivity/summary/{d}|all/sens_{rule}'):.3f}", f"sensitivity/summary/{d}|all/sens_{rule}", "{:.3f}"))
for h in hs:
    for f in ("after", "no_prior"):
        CHK.append(("F2 (range ends only)", f"{get(f'masking/R/{h}/{f}'):.3f}", f"masking/R/{h}/{f}", "{:.3f}"))
F2_nums = set(re.findall(r"\d+\.\d+", F2))
SENT_ROWS = []
for where, shown, key, fmt in CHK:
    v = get(key)
    val = (fmt[5:].format(100 * v) if fmt.startswith("x100:") else fmt.format(v))
    assert val == shown, (where, shown, val)
    used = where != "F2 (range ends only)" or shown in F2_nums
    if used:
        SENT_ROWS.append((where, shown, key))
for s in ("0.192", "0.261", "0.272", "0.391"):
    assert s in F2_nums

PLAN_CONST = [
    ("15 PPG-DaLiA subjects; 16 WildPPG participants; route Zurich → Jungfraujoch, up to 3,571 m", "prereg v2.2 §0 (L1), §1; NOTES_for_paper.md §5"),
    ("8 s windows, 2 s stride; L = 10 min = 300 windows; 6 blocks per monitored hour", "prereg v2.2 §2; R1.md §1"),
    ("PPG 128 → 64 Hz, ACC 128 → 32 Hz", "prereg v2.2 §1"),
    ("σ = 3 bpm; seeds 0–4; fine-tune gain 0.57 bpm vs 1.0 bpm rule", "prereg v2.2 §1"),
    ("motion state edges 0.02 g and 0.10 g", "prereg v2.2 §1"),
    ("K: 6 calibration blocks, test ≥ 6; K m = 1: blocks 0,2,…,10, buffer 1, test ≥ 13; R: j−7..j−2, test ≥ 7; R-thin: odd lags 3–13", "prereg v2.2 §2"),
    ("ACI target 1/7, step 0.05; B0 = 120 bpm; fallback 120 bpm", "prereg v2.2 §2, §3"),
    ("bootstrap 10,000 draws; p_order B = 2,000; covariate pairs bootstrap 2,000 draws, 90 % CI; drop rule abs(coef) > 20, 5 % limit", "prereg v2.2 §4.0–§4.2"),
    ("H-K1 simulated size 0.045–0.051 (iid / MA(1) 0.17 / 0.30)", "prereg v2.2 §4.1b table"),
    ("injected Δ +20/+30/+40 bpm, ramp 30 s, plateau 180 s, 3 per hour", "prereg v2.2 §4.4"),
    ("MA(1) table in Appendix A (0.1426 … 1.092)", "03_theory/R1.md §4b table"),
    ("SHA256 35474f6a…dfd7b4; commits 2decaae, d402033", "DEVIATIONS.md header"),
]

# ---------------------------------------------------------------- 5. T6 (microcontroller; development participants, seed 0)
# Source: 05_experiments/t6_mcu/results/ (REPORT_T6 commit 7508dab) and explore_input_hash.json (exploratory, S3d).
# Not from results.json and not from the 13 held-out persons. Operations: rounding, percentiles of stored board timings,
# sums and ratios of stored counts, as stated in the source column.
import numpy as np  # noqa: E402
T6R = ROOT / "05_experiments/t6_mcu/results"; T6M = ROOT / "05_experiments/t6_mcu/model"
T6ROWS = []
J6 = lambda f: json.load(open(T6R / f))
DEVIDS = ("e61", "u7y", "ngh")


def t6(macro, s, src):
    T6ROWS.append((clean(macro), str(s), src))
    return s


c3 = lambda x: f"{x:,}"
Bd = np.load(T6R / "B_s3_dense_runs.npz")["t_us"] / 1000; Bo = np.load(T6R / "B_s3_orig_runs.npz")["t_us"] / 1000
Bdi, Boi = J6("B_s3_dense_info.json"), J6("B_s3_orig_info.json")
NPZ = "`B_s3_dense_runs.npz` `t_us` (runs × windows × stage; ms = µs/1000; stage 4 = total, 1 = inference)"
NPO = "`B_s3_orig_runs.npz` `t_us` (stage 4 = total)"
t6("tsBRuns", Bd.shape[0], NPZ + ", axis 0 length")
t6("tsBWinRun", c3(Bd.shape[1]), NPZ + ", axis 1 length")
t6("tsBWinPerson", c3(Bd.shape[1] // 3), NPZ + ", axis 1 length / 3 persons (`replay.py --b-windows 1800`)")
t6("tsBWinAll", c3(Bd.shape[0] * Bd.shape[1]), NPZ + ", runs × windows")
t6("tsBoRuns", Bo.shape[0], NPO + ", axis 0 length"); t6("tsBoWin", Bo.shape[1], NPO + ", axis 1 length")
t6("tsTotMed", f"{np.percentile(Bd[..., 4], 50):.2f}", NPZ + ", 50th percentile of total over all runs and windows")
t6("tsTotPnn", f"{np.percentile(Bd[..., 4], 99):.2f}", NPZ + ", 99th percentile of total")
t6("tsTotPnnInt", f"{np.percentile(Bd[..., 4], 99):.0f}", NPZ + ", 99th percentile of total, rounded to 1 ms")
t6("tsTotMax", f"{Bd[..., 4].max():.2f}", NPZ + ", maximum of total")
t6("tsInfMed", f"{np.percentile(Bd[..., 1], 50):.2f}", NPZ + ", 50th percentile of inference")
t6("tsMaxFirst", int(np.sum(Bd[..., 4].argmax(1) == 0)), NPZ + ", runs whose maximum total is window 0 (first after reset)")
t6("tsRatioPct", f"{100 * np.percentile(Bd[..., 4], 99) / 2000:.1f}", NPZ + ", 100 × p99(total) / 2,000 ms stride")
t6("tsOrigPnn", f"{np.percentile(Bo[..., 4], 99):.0f}", NPO + ", 99th percentile of total, rounded to 1 ms")
t6("tsOrigRatioPct", f"{100 * np.percentile(Bo[..., 4], 99) / 2000:.1f}", NPO + ", 100 × p99(total) / 2,000 ms")
sz = (T6R / "fw_size/size_s3_dense.txt").read_text()
t6("tsImageKB", f"{int(re.search(r'Total image size: (\d+)', sz).group(1)) / 1000:.0f}", "`fw_size/size_s3_dense.txt` 'Total image size' / 1000")
t6("tsArenaKB", f"{Bdi[0]['end']['arena_used'] / 1000:.1f}", "`B_s3_dense_info.json` `0/end/arena_used` / 1000 (equal in all runs, REPORT_T6 §4.1)")
assert len({b["end"]["arena_used"] for b in Bdi}) == 1 and len({b["end"]["min_free_heap"] for b in Bdi}) == 1
t6("tsArenaResKB", f"{Bdi[0]['info']['arena_bytes'] / 1000:.1f}", "`B_s3_dense_info.json` `0/info/arena_bytes` / 1000")
t6("tsHeapKB", f"{Bdi[0]['end']['min_free_heap'] / 1000:.0f}", "`B_s3_dense_info.json` `0/end/min_free_heap` / 1000")
t6("tsArenaEspKB", f"{J6('B_ref_esp32_info.json')[0]['end']['arena_used'] / 1000:.1f}", "`B_ref_esp32_info.json` `0/end/arena_used` / 1000 (ESP32-D0WD-V3)")
t6("tsMHz", Bdi[0]["info"]["cpu_mhz"], "`B_s3_dense_info.json` `0/info/cpu_mhz`")
t6("tsIdf", Bdi[0]["info"]["idf"], "`B_s3_dense_info.json` `0/info/idf`")
dep = (ROOT / "06_firmware/h32_e2/dependencies.lock").read_text()
ver = dict(re.findall(r"espressif/(esp-[a-z-]+):\n(?:.*\n)*?    version: ([0-9.]+)", dep))
t6("tsTflm", ver["esp-tflite-micro"], "`06_firmware/h32_e2/dependencies.lock` espressif/esp-tflite-micro version")
t6("tsEspnn", ver["esp-nn"], "`06_firmware/h32_e2/dependencies.lock` espressif/esp-nn version")
qm6 = json.load(open(T6M / "quant_meta.json")); qd6 = json.load(open(T6M / "quant_meta_dense.json"))
t6("tsCalib", qm6["calib_windows"], "`05_experiments/t6_mcu/model/quant_meta.json` `calib_windows`")
assert all(v["c_core_inputs"] == v["n"] for v in qd6["identical_windows"].values())
t6("tsDenseEqPC", c3(sum(v["n"] for v in qd6["identical_windows"].values())), "`model/quant_meta_dense.json` `identical_windows/*/c_core_inputs` (sum; all = n)")
A6 = J6("A_board.json"); S6 = J6("A_S3_reduced.json")
t6("tsAWin", c3(sum(A6["persons"][s]["n_windows_device"] for s in DEVIDS)), "`A_board.json` `persons/*/n_windows_device` (sum)")
for s in DEVIDS:
    t6(f"tsHashEq{s}", c3(A6["persons"][s]["input_hash_equal_board_vs_pc_ccore"]), f"`A_board.json` `persons/{s}/input_hash_equal_board_vs_pc_ccore`")
    a_, b_ = map(int, A6["persons"][s]["logits_bit_identical_given_equal_input_hash"].split("/")); assert a_ == b_
assert all(S6["models"][m]["persons"][s]["vs_pc_int8"]["logits_identical_given_equal_hash"].split("/")[0] ==
           S6["models"][m]["persons"][s]["vs_pc_int8"]["logits_identical_given_equal_hash"].split("/")[1] for m in ("dense", "orig") for s in DEVIDS)
t6("tsSthreeHashEq", c3(sum(S6["models"]["dense"]["persons"][s]["vs_pc_int8"]["input_hash_equal"] for s in DEVIDS)),
   "`A_S3_reduced.json` `models/dense/persons/*/vs_pc_int8/input_hash_equal` (sum; logits equal on all of them)")
blk = [A6["persons"][s]["device_block_code_vs_python_on_device_outputs"] for s in DEVIDS]; assert all(b["mismatches"] == 0 for b in blk)
t6("tsBlocks", sum(b["records"] for b in blk), "`A_board.json` `persons/*/device_block_code_vs_python_on_device_outputs/records` (sum; mismatches 0)")
t6("tsSthreeWin", c3(int(S6["models"]["dense"]["vs_esp32_all_five_total"].split("/")[1])), "`A_S3_reduced.json` `models/dense/vs_esp32_all_five_total` (denominator)")
assert S6["models"]["dense"]["reduced_A_stop_condition_met"] and S6["models"]["orig"]["vs_esp32_all_five_total"] == S6["models"]["dense"]["vs_esp32_all_five_total"]
for r in ("K", "R"):
    t6(f"ts{r}Diff", A6["pooled"][r]["decisions_differ"], f"`A_board.json` `pooled/{r}/decisions_differ`")
    t6(f"ts{r}Test", A6["pooled"][r]["test_blocks"], f"`A_board.json` `pooled/{r}/test_blocks`")
E6 = J6("A_emul.json")
assert all(E6["persons"][s][r]["decisions_differ"] == A6["persons"][s][r]["decisions_differ"] for s in DEVIDS for r in "KR")
t6("tsKRPct", f"{100 * (A6['pooled']['K']['decisions_differ'] + A6['pooled']['R']['decisions_differ']) / (A6['pooled']['K']['test_blocks'] + A6['pooled']['R']['test_blocks']):.1f}",
   "`A_board.json` 100 × (K + R decisions_differ) / (K + R test_blocks)")
p99s = [A6["persons"][s]["dHR_all"]["p99"] for s in DEVIDS]
t6("tsHRpnnLo", f"{min(p99s):.1f}", "`A_board.json` `persons/*/dHR_all/p99` (min)"); t6("tsHRpnnHi", f"{max(p99s):.1f}", "… (max)")
t6("tsHRmax", f"{max(A6['persons'][s]['dHR_all']['max'] for s in DEVIDS):.0f}", "`A_board.json` `persons/*/dHR_all/max` (max)")
assert all(A6["persons"][s]["dHR_all"]["median"] == 0 for s in DEVIDS)
fl = [100 * A6["persons"][s]["monitored_flag_differs"] / A6["persons"][s]["n_windows_device"] for s in DEVIDS]
t6("tsFlagLo", f"{min(fl):.1f}", "`A_board.json` 100 × `persons/*/monitored_flag_differs` / `n_windows_device` (min)"); t6("tsFlagHi", f"{max(fl):.1f}", "… (max)")
t6("tsStateDiff", sum(A6["persons"][s]["state_differs"] for s in DEVIDS), "`A_board.json` `persons/*/state_differs` (sum)")
cells = [(s, k, v) for s in DEVIDS for k, v in sorted(A6["persons"][s]["K"]["tau_per_state"].items())]
chg = [(s, k, v) for s, k, v in cells if v["float"] != v["device"]]
t6("tsTauCells", len(cells), "`A_board.json` `persons/*/K/tau_per_state` (count of cells)")
t6("tsTauChanged", len(chg), "… cells with float ≠ device")
SN = {"0": "rest", "1": "light", "2": "vigorous"}
t6("tsTauList", "; ".join(f"{s}, {SN[k]}: {v['float']:.1f} to {v['device']:.1f} bpm" for s, k, v in chg), "… those cells, float → device")
X6 = J6("explore_input_hash.json"); rp = X6["recovery_from_hash"]["persons"]
ndiff = sum(p["windows_hash_differs"] for p in rp.values()); nall = sum(A6["persons"][s]["n_windows_device"] for s in DEVIDS)
t6("tsHashDiff", c3(ndiff), "`explore_input_hash.json` `recovery_from_hash/persons/*/windows_hash_differs` (sum; = n_windows − hash-equal in A_board.json)")
assert ndiff == nall - sum(A6["persons"][s]["input_hash_equal_board_vs_pc_ccore"] for s in DEVIDS)
t6("tsHashDiffPct", f"{100 * ndiff / nall:.1f}", "100 × tsHashDiff / tsAWin")
t6("tsHashDiffSthree", c3(sum(int(p["recovered_first_1800"].split("/")[1]) for p in rp.values())), "`explore_input_hash.json` `…/recovered_first_1800` (denominators, sum)")
nrec = sum(p["recovered"] for p in rp.values())
t6("tsRecPct", f"{100 * nrec / ndiff:.1f}", "`explore_input_hash.json` 100 × sum `recovered` / sum `windows_hash_differs`")
assert all(p["max_abs_int8_diff"] == 1 for p in rp.values())
t6("tsRecBinsMax", max(int(k) for p in rp.values() for k, v in p["bins_differ_counts"].items() if v), "`explore_input_hash.json` `…/bins_differ_counts` (largest non-empty)")
t6("tsRecDist", f"{max(p['max_distance_of_changed_value_from_boundary'] for p in rp.values()):.3f}", "`explore_input_hash.json` `…/max_distance_of_changed_value_from_boundary` (max, quantisation steps)")
assert X6["variant_reproducing_board"] is None

# ---------------------------------------------------------------- write
(OUT / "numbers.tex").write_text("% generated by 07_manuscript/make_numbers.py; do not edit\n" +
                                 "\n".join(f"\\newcommand{{\\{m}}}{{{tex(v)}}}" for m, v, _ in ROWS + T6ROWS) + "\n")
md = ["# H32 · NUMBERS: every number of the manuscript and its source", "",
      "Generated by `07_manuscript/make_numbers.py`. `results.json` = `05_experiments/confirm/runs/confirmatory/results.json`,",
      f"SHA256 `{SHA}` (checked by the script). Keys are paths inside that file (`a/b/0` = list element 0).",
      "Numbers are rounded only; min–max rows take the minimum and maximum over values that are themselves in the file.",
      "No number is computed from the 13 persons' data.", "",
      "## 1. Numbers inside the fixed sentences (verbatim from conclusions.md and DEVIATIONS.md)", "",
      "| Sentence | Number as printed | Key in results.json |", "|---|---|---|"]
md += [f"| {w} | {s} | `{k}` |" for w, s, k in SENT_ROWS]
md += ["", "Rounding of every row above was checked by the script (assert).", "",
       "## 2. Macros used in main.tex", "", "| Macro | Value | Source |", "|---|---|---|"]
md += [f"| `\\{m}` | {v} | {src} |" for m, v, src in ROWS]
md += ["", "## 3. Tables and figures", "",
       "Tables T1–T4 and A2–A3 are written by `07_manuscript/make_tables.py`, figures F1–F4 by",
       "`07_manuscript/figs/make_figs.py`; each script names the keys it reads in its docstring.", ""]
aux = HERE_AUX = ROOT / "07_manuscript/main.aux"
labels = dict(re.findall(r"newlabel\{((?:tab|fig):[a-z0-9]+)\}\{\{(\d+)\}", aux.read_text())) if aux.exists() else {}
MAP = [("T1", "tab:far", "FAR and coverage per rule"), ("T2", "tab:hk", "per person, H-K and H-K1"),
       ("T3", "tab:hr", "H-R, R-thin, paired, ACF"), ("T4", "tab:dev", "deviations D1–D3 and disclosures"),
       ("A2", "tab:cells", "per cell"), ("Amask", "tab:masking", "masking with n per rule"),
       ("T5", "tab:mcu", "T6 microcontroller: time, memory, Measurement A"),
       ("F1", "fig:cells", "K rate per cell, X per seed"), ("F2", "fig:dalt", "exceedance by dAlt tercile"),
       ("F3", "fig:cost", "sensitivity and masking"), ("F4", "fig:coverage", "waiting time and wear covered")]
md += ["## 3b. Internal labels → numbers in main.pdf (read from main.aux of the last build)", "",
       "| Internal | LaTeX label | Printed as | Content |", "|---|---|---|---|"]
md += [f"| {i} | `{l}` | {('Table ' if l.startswith('tab') else 'Figure ') + labels.get(l, '?')} | {c} |" for i, l, c in MAP]
md += ["", "Table 1 is the deviation/disclosure table (T4) because Methods cites it first (PI decision B7, 2026-11-23).", ""]
md += ["## 4. Constants of the fixed plan and of the datasets, typed in main.tex (not results)", "",
       "| Value in main.tex | Source |", "|---|---|"]
md += [f"| {v} | {src} |" for v, src in PLAN_CONST]
md += ["", "## 5. T6: microcontroller (development participants e61, u7y, ngh; E2 seed 0; not confirmatory)", "",
       "Files in `05_experiments/t6_mcu/results/` unless a path is given (REPORT_T6, commit 7508dab; `explore_input_hash.json`",
       "is exploratory, prompt S3d). Not from results.json and not from the 13 held-out persons. Table T5 is written by",
       "`make_tables.py` from the same files.", "", "| Macro | Value | Source (file, key, operation) |", "|---|---|---|"]
md += [f"| `\\{m}` | {v} | {src} |" for m, v, src in T6ROWS]
(ROOT / "07_manuscript/NUMBERS.md").write_text("\n".join(md) + "\n")
print(len(ROWS), "macros;", len(SENT_ROWS), "sentence numbers checked")
