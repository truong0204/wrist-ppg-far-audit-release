"""H32 S3a: figures F1–F4 from results.json only (PDF + PNG). One message per figure; rates in /monitored h;
the ratio 8.80/4.39 is never plotted.
  F1  K rate per (person, state) cell vs r = 2/monitored h, and X per E2 seed vs E0   keys FAR/K/per_cell, H_K/X1
  F2  exceedance rate of K test blocks per dAlt tercile                               keys covariates/exceed_by_dalt_tercile, tercile_edges_km
  F3  cost of rolling calibration: sensitivity by Δ, and masking by lag h             keys sensitivity/summary/<Δ>|all, masking/<rule>/<h>
  F4  wear time before a within-person threshold is active                            keys coverage/cells, coverage/per_person_frac_wear
Palette: reference categorical slots 1–3 (validated all-pairs, light mode), neutral greys for reference lines.
    uv run python 07_manuscript/figs/make_figs.py
"""
import hashlib, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = pathlib.Path(__file__).parent
raw = (ROOT / "05_experiments/confirm/runs/confirmatory/results.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "2460459493a626f8e6205a1fbdc1080278378e990ec8c5d5ac013cfee77cffb2"
R = json.loads(raw); IDS = R["ids"]; STATES = ("rest", "light", "vigorous")
C = {"rest": "#2a78d6", "light": "#eb6834", "vigorous": "#1baf7a"}          # slots 1-3
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)

# ------------------------------------------------------------------ F1
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [3.2, 1]})
pc = R["FAR"]["K"]["per_cell"]; off = {"rest": -0.22, "light": 0.0, "vigorous": 0.22}
for st in STATES:
    xs, ys = [], []
    for i, s in enumerate(IDS):
        c = pc.get(f"{s}|{st}")
        if c:
            xs.append(i + off[st]); ys.append(c["rate"])
    a.scatter(xs, ys, s=16, color=C[st], edgecolor="white", linewidth=0.6, label=st, zorder=3)
a.axhline(2, color=INK2, lw=1.0, ls="--", zorder=2); a.text(-0.5, 2.06, "r = 2", ha="left", va="bottom", color=INK2)
a.set_xticks(range(len(IDS))); a.set_xticklabels(IDS, rotation=0); a.set_xlim(-0.6, len(IDS) - 0.4)
a.set_ylabel("False alarms (/monitored h)"); a.set_xlabel("Held-out person")
a.set_title("K: rate per (person, state) cell, seed average"); grid(a)
a.set_ylim(-0.2, 4.6); a.legend(title="Motion state", ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.07), fontsize=7, title_fontsize=7, handletextpad=0.2, columnspacing=0.8)
X1 = R["H_K"]["X1"]
seeds = [r["seed"] for r in X1]
b.plot(seeds, [r["X"] for r in X1], "o", color=INK, ms=4.5, label="observed X", zorder=3)
b.plot(seeds, [r["E0"] for r in X1], "_", color=INK2, ms=12, mew=1.6, label="E0 under A(0)", zorder=3)
b.set_xticks(seeds); b.set_xlabel("E2 seed"); b.set_ylabel("Cells above r (of 39)"); b.set_ylim(0, 12.5)
b.set_title("Cells above r, per seed"); grid(b); b.legend(loc="lower left", fontsize=7, handletextpad=0.3)
fig.tight_layout(); save(fig, "F1_cells_K")

# ------------------------------------------------------------------ F2
cov = R["covariates"]; e = cov["tercile_edges_km"]; ex = cov["exceed_by_dalt_tercile"]
labels = [f"low\n< {e[0]:.2f} km", f"mid\n{e[0]:.2f}–{e[1]:.2f} km", f"high\n> {e[1]:.2f} km"]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.bar(range(3), ex, width=0.55, color="#2a78d6", edgecolor="white", linewidth=2)
for i, v in enumerate(ex):
    ax.text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom", color=INK)
ax.axhline(1 / 7, color=REF, lw=1.0, ls=":"); ax.text(2.45, 1 / 7 + 0.006, "1/7", ha="right", va="bottom", color=INK2, fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylim(0, 0.42)
ax.set_ylabel("Fraction of K test blocks\nwith an exceedance"); ax.set_xlabel("Altitude change since calibration (dAlt tercile)")
ax.set_title("K exceedances by altitude change"); grid(ax)
fig.tight_layout(); save(fig, "F2_dalt_tercile")

# ------------------------------------------------------------------ F3
S = R["sensitivity"]["summary"]; D = (20, 30, 40)
RULE_C = {"K": "#2a78d6", "R": "#eb6834", "B0": "#1baf7a"}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for rule, lab in (("K", "K"), ("R", "R"), ("B0", "B0 (120 bpm)")):
    ys = [S[f"{d}|all"][f"sens_{rule}"] for d in D]
    a.plot(D, ys, "-o", color=RULE_C[rule], lw=2, ms=5, mec="white", mew=1, label=lab)
    a.text(D[-1] + 1.2, ys[-1], lab.split(" ")[0], va="center", color=INK, fontsize=7)
a.set_xticks(D); a.set_xlim(17, 45); a.set_ylim(0, 0.6); a.set_xlabel("Injected HR excursion Δ (bpm)")
a.set_ylabel("Sensitivity (events detected)"); a.set_title("Detection of injected events"); grid(a)
a.legend(loc="upper left", fontsize=7)
M = R["masking"]; hs = [str(h) for h in range(1, 15)]
b.plot(range(1, 15), [M["R"][h]["no_prior"] for h in hs], "-", color=RULE_C["R"], lw=1.2, alpha=0.55, label="R, no prior event")
b.plot(range(1, 15), [M["R"][h]["after"] for h in hs], "-o", color=RULE_C["R"], lw=2, ms=4, mec="white", mew=0.8, label="R, after a first event")
b.plot(range(1, 15), [M["K"][h]["after"] for h in hs], "-", color=RULE_C["K"], lw=2, label="K (with or without)")
b.axvspan(1.5, 7.5, color=GRID, alpha=0.6, lw=0); b.text(4.5, 0.03, "lags 2–7: in R's window", ha="center", color=INK2, fontsize=7)
b.set_xlim(0.5, 14.5); b.set_ylim(0, 0.85); b.set_xlabel("Second event h state-blocks after the first (Δ = +30 bpm)")
b.set_ylabel("Detection of the second event"); b.set_title("Masking after an event (block-level simulation)"); grid(b)
b.legend(loc="upper right", fontsize=6.5, ncol=1, bbox_to_anchor=(1.0, 1.02))
fig.tight_layout(); save(fig, "F3_cost_R")

# ------------------------------------------------------------------ F4
cells = R["coverage"]["cells"]; fw = R["coverage"]["per_person_frac_wear"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
for j, st in enumerate(STATES):
    v = [c["K_cal_done_h"] for c in cells if c["state"] == st and c["K_cal_done_h"] is not None]
    a.scatter([j + (0.07 if k % 2 else -0.07) for k in range(len(v))], sorted(v), s=14, color=C[st], edgecolor="white", linewidth=0.5, zorder=3)
a.set_xticks(range(3)); a.set_xticklabels(STATES); a.set_xlim(-0.6, 2.6)
a.set_ylabel("Hours until K is active (real time)"); a.set_title("Waiting time per (person, state) cell"); grid(a)
order = sorted(IDS, key=lambda s: fw[s]["K"])
for k, (rule, lab, col) in enumerate((("K", "K", "#2a78d6"), ("R", "R", "#eb6834"), ("Rthin", "R-thin", "#1baf7a"))):
    b.scatter([fw[s][rule] for s in order], range(len(order)), s=16, color=col, edgecolor="white", linewidth=0.6, label=lab, zorder=3)
b.set_yticks(range(len(order))); b.set_yticklabels(order); b.set_xlim(0, 1)
b.set_xlabel("Fraction of wear time with an active threshold"); b.set_title("Wear time covered, per person"); grid(b, "x")
b.legend(loc="lower right", fontsize=7)
fig.tight_layout(); save(fig, "F4_coverage")
print("figures:", sorted(p.name for p in OUT.glob("F*.p*")))
