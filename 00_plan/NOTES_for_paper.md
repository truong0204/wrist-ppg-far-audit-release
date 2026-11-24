# H32 · Notes for the paper (not deviations)

From numbers already in `05_experiments/confirm/runs/confirmatory/results.json` (SHA256 `24604594…cffb2`) and in
prereg v2.2. Nothing below is computed from the 13 persons' data. Written 2026-11-23 (prompt S1m).

## 1. H-K per E2 seed (X1)

| E2 seed | X (cells above r = 2/h, of 39) | E0 | X1 p (exact Poisson-binomial) |
|---|---|---|---|
| 0 | 9 | 4.48 | 0.029 |
| 1 | 11 | 4.37 | 0.003 |
| 2 | 10 | 4.36 | 0.009 |
| 3 | 5 | 4.35 | 0.442 |
| 4 | 9 | 4.38 | 0.026 |

Source: `H_K.X1`. Seed average X̄ = 8.80 (`H_K.Xbar`); median p = 0.026 (`H_K.X1_median_p`).

- The person-cluster bootstrap resamples persons on values already averaged over the 5 seeds (§4.0, §4.1:
  d_i = mean over seeds). Seeds are not resampled, so the CI does not contain the variation between E2 seeds.
  The table above is the only view of that variation and should be in the paper.

## 2. Size of X2 at the observed lag-1 dependence

- Observed lag-1 autocorrelation of block scores, pooled (`H_R.acf`): 0.20 raw [0.15, 0.26], 0.15 after a linear
  detrend [0.09, 0.21] (90 % CI, Fisher z).
- Simulated size of X2 (prereg §4.1b table, 13 persons, dev block profiles): 0.096 at lag-1 = 0.17 and 0.166 at
  lag-1 = 0.30 (stationary MA(1)). At the observed lag-1 the size of X2 is therefore about 0.10, not 0.05.
- H-K1 keeps its level in the same simulation (0.046 at 0.17; 0.045 at 0.30). This is why the fixed sequence
  H-K → H-K1 matters for the wording.

## 3. H-K1 rests partly on one person and on short cells

- Σ d_i = 28.06 (`H_K1.sum_d`); qm9 contributes 10.60 (`H_K1.d_i.qm9`).
- K with m = 1 has test blocks in 33 of the 39 cells (`FAR.Km1.per_cell`); 9 of these 33 cells have T ≤ 3 test
  blocks (seed mean); the smallest is T = 1.
- The bootstrap 5 % quantile (6.54) already reflects this. A leave-one-person-out view is not in the plan; if the
  PI asks for it, it is labelled "exploratory, after unblinding".

## 4. Covariate definitions (prereg §4.2 d)

- dAlt = |mean altitude of the block − mean altitude of the 6 calibration blocks|, in km (**absolute value**).
- dTemp = the same for the device-internal wrist temperature, in °C (**absolute value**).
- dTime = mean real time of the block − mean real time of the 6 calibration blocks, in hours (**signed**).
- Altitude is barometric (variable `data_altitude_values` of the authors' `WildPPG.mat`), one value per 8 s;
  0 is treated as missing. Before opening, only length and missing fraction were inspected for the confirmatory
  IDs: present for all 16 IDs, no NaN, value 0 in 5.8 % (k2s) and 3.4 % (l38) of samples, otherwise 0 %.
- Temperature is "temperature within device" (authors' description): neither skin nor ambient.

## 5. Data, training and leakage (answers to S1m item 3, from the repo)

**E2 training.**
- E2 = SpecFCN + causal HMM (σ = 3 bpm). Final weights `fcn_all_seed{0..4}.pt` were trained on **all 15 PPG-DaLiA
  subjects** (`05_experiments/s1c/train_all_dalia.py`, commit `32704da`, 2026-11-21; docstring: "No WildPPG data is
  read here"). Prereg v2.1 (`9329ba2`) made DaLiA training/dev only and WildPPG the sole confirmatory set.
- Architecture and σ were chosen on DaLiA: LOSO dev folds S10–S12 (`eval_dev.py`, `44492f1`), σ grid extended
  on DaLiA dev, LOO picks 2–3 (`sigma_grid_ext.py`, `7535428`); σ = 3 fixed (prereg v2.2 §1, deviation row 8).

**WildPPG choices made on the 3 dev IDs only.**
- Zero-shot vs fine-tune: decided on e61/u7y/ngh (`wildppg_dev_check.py`, rule fixed in `06ab627`, result
  `46c0b7a`: gain 0.57 < 1.0 bpm → keep zero-shot).
- SQI gate θ = 0.0290: smallest passing quantile q = 0.15 of SQI pooled over the 3 dev IDs × 5 seeds
  (`sqi_dev.py`, rule fixed in `b4242f7` before running, result `8afbb7b`).
- Block rules, m, dev controls, power simulations: dev IDs and simulations only (prereg v2.2 §5).

**Did any of the 13 confirmatory persons enter training, hyperparameter choice or the SQI gate?** No
confirmatory **PPG** was read before opening: `io_wildppg.load_wrist` raised `PermissionError` for the 13 IDs
while `CONFIRMATORY_PPG_OPEN = False` (opened only in `d402033`). What was read of the confirmatory IDs before
opening, as allowed by prereg v2 §2 item 2:
- reference HR labels (authors' cleaned HR, `data_bpm_values`), for label-only counts and recording lengths
  (`05_experiments/s1b/B_wildppg.py`, `C4_vacuity.py`; commit `2174788`). **Correction (2026-11-23, S1n):** an
  earlier version of this line said "and ACC". That was wrong for WildPPG: the wrist ACC of the 13 persons is read
  only by `io_wildppg.load_wrist`, which was locked until `d402033`. ACC was read for DaLiA (`B_counts.py`), see §6;
- to map the HF file to IDs, the per-ID `sternum` struct was loaded to read the ECG length; that struct also holds
  sternum PPG arrays, which were discarded unread (documented in the prereg v2 change log). Wrist PPG was not
  loaded;
- length and missing fraction of the altitude and temperature channels, not their values. This is not covered by
  prereg v2 §2 item 2 (labels and ACC only); it is recorded in prereg v2.2 §4.2 d ("only their length and missing
  fraction were inspected", 2026-11-22), before freezing.

**Dev/confirmatory split of WildPPG.**
- Rule: the 3 IDs with the smallest SHA256 of the string `WildPPG_Part_<ID>`, over the 16 IDs of the public share
  listing (file names only), fixed before any WildPPG file was downloaded (prereg v2 §1, commit `e873b1e`,
  2026-11-21 14:58 +07). The code constant `DEV = ("e61", "u7y", "ngh")` is in `04_code/h32/io_wildppg.py`.
- Re-checked from the repo: the three smallest hashes are e61 (`0a4eda2b…`), u7y (`10a2ce07…`), ngh
  (`1cf26170…`); the next is p9p (`207b0c03…`).

## 6. Disclosures for Methods

What was read of the confirmatory data before opening (`d402033`, 2026-11-23 13:23:33 +07). WildPPG wrist PPG and
wrist ACC of the 13 confirmatory persons were **not** read: `io_wildppg.load_wrist` (the only code path to the
`wrist` struct) raised `PermissionError` for them while `CONFIRMATORY_PPG_OPEN = False`.

| What | Of whom | Why | Script | Commit |
|---|---|---|---|---|
| **Values** of the reference HR labels (authors' cleaned HR, `data_bpm_values` of the HF file): `hours_valid`, `frac_removed`, number of episodes HR > 100 / 110 / 120 per person and per group | all 16 WildPPG IDs, the 13 confirmatory persons included | label-only counts of reference episodes | `05_experiments/s1b/B_wildppg.py` → `results/B_wildppg.json`, printed to screen | `2174788` (2026-11-21 15:38 +07) |
| **Length only** of the reference HR series (`data_bpm_values`): hours per HF index, and test hours left per (L, r) as median and minimum over the 16 | all 16 WildPPG HF indices (not mapped to IDs; dev persons included) | vacuity table C4 (test time left) | `05_experiments/s1b/C4_vacuity.py` → `results/C4_vacuity.json`, printed; medians in the C4 table of `05_experiments/s1b/C_report.md` | `2174788` |
| Length of the sternum ECG (`sternum.ecg.v`) | all 16 WildPPG IDs | to map the HF file (no IDs) to IDs by recording length; the `sternum` struct also holds sternum PPG arrays, discarded unread | `05_experiments/s1b/B_wildppg.py` | `2174788`; recorded in the prereg v2 change log |
| Length and missing fraction of altitude and wrist temperature (`data_altitude_values`, `data_temp_wrist`), not their values | all 16 WildPPG IDs | to check that the exogenous covariates exist before freezing | no script in the repo; result recorded in the commit message and in prereg v2.2 §4.2 d | `d43dc75` (2026-11-22 21:00 +07) |
| HR-ECG labels and wrist ACC | all 15 PPG-DaLiA subjects (S1–S9, S13–S15 were confirmatory under v2) | label-only counts (B counts, C4) | `05_experiments/s1b/B_counts.py`, `C4_vacuity.py` | `2174788`; DaLiA became training/dev only in v2.1 (`9329ba2`), so it is not part of the confirmatory set |

- The altitude/temperature inspection is outside the scope of prereg v2 §2 item 2 ("reference labels (HR-ECG) and
  accelerometer only"); it is recorded in v2.2 §4.2 d before freezing.
- Block length L = 10 min and r = 2/h were fixed in prereg v2.1 (`9329ba2`, 2026-11-21 15:54 +07, change row 4:
  "L = 10 min, r = 2/h primary | S1b C4"). The deciding criterion was the calibration time after the ρ-adjustment
  ("the only configuration ≤ 60 min after the ρ-adjustment is L = 10, r = 2", v2.1 §4), with ρ from the DaLiA dev
  persons; it does not use WildPPG data. The WildPPG test hours (from the length of the HR series of all 16 indices,
  confirmatory persons included) stand in the same table; they were used for another decision: WildPPG as the sole
  confirmatory set and DaLiA as training/dev ("WildPPG leaves ≈ 9–12 h", v2.1 §1). The per-state calibration with
  6 blocks followed in the v2.2 draft (`b88e4e3`, 2026-11-22). The WildPPG confirmatory ACC, and therefore the
  states and the number of blocks per cell of the 13 persons, were **not** known before opening.
- `B_wildppg.py` (`2174788`) computed from the **values** of the reference HR labels of the 13 confirmatory persons
  `hours_valid`, `frac_removed` and the number of episodes HR > 100 / 110 / 120 (group `confirm13`: 809 / 471 / 185;
  13 of 13 persons with ≥ 1 episode at each threshold), printed them to screen and saved them to
  `05_experiments/s1b/results/B_wildppg.json`. The decision to use the in-control framing with injected events
  (v2.1, change row 2: "S1b B: D2 ≈ 0, D3 circular or protocol-driven") cites the source "S1b B" without separating
  DaLiA from WildPPG. An influence of these numbers on that decision can therefore not be excluded.

**Note for the PI.** The S1n prompt said that the ACC of the 13 persons was read for counting and that block
length and calibration size were therefore set with part of the confirmatory cell structure known. The repo does
not support this for WildPPG (see the table); the premise came from the wrong "and ACC" in §5, now corrected. The
bullets above state what the repo shows.

