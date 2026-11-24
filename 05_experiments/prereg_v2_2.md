# H32 · Pre-registration v2.2 (frozen)

Date: 2026-11-23 (revision 8, frozen on 2026-11-23 after the PI wrote "duyệt đóng băng").
- Revision 2 followed the PI decision `00_plan/PI_DECISION_S2r1_2026-11-22.md`.
- Revision 3 follows the Cowork proof check `00_plan/REVIEW_S2_r3_proofcheck_2026-11-22.md` (prompt S1f: P1–P3, Q1–Q6).
- Revision 4 applies the round-2 items S1, S2, S4, L1 and L2 (`REVIEW_S2_r2_2026-11-22.md`; S3 was replaced by Q2 of
  round 3), the PI checks B1–B3 on R / R-thin, and the new citation rule (PI prompt after S1f).
- Revision 5 follows the PI prompt after d43dc75 and `00_plan/REVIEW_S2_r4_proofcheck2_2026-11-22.md`:
  - p_order wording;
  - altitude model with elapsed time;
  - refs cross-check column;
  - clean-up C1–C3 and D2–D4.
- Revision 6 (PI prompt after 7cd7a1a) makes K with m = 1 the second test of a fixed sequence (H-K1, §4.1b), and adds
  the rule for bootstrap draws in which the logistic fit fails (§4.2 d). Approved by the PI with the |coefficient| > 20
  separation threshold.
- Revision 7 (PI): the "H-K not rejected" sentence always carries the expectation-level result (Y2), with its own
  branch (§4.1); §4.7 records the confirmatory script, its dry run on the dev IDs, and the environment.
- Revision 8 (PI 23-11) adds operating rules only; no test is changed:
  - B0-state is deleted (§3, deviation rows 3 and 20);
  - rules for persons and cells with missing data (§4.0);
  - what "run once" means (§4.7);
  - §7 items 1–4 approved by the PI.

Builds on, all unchanged:
- `prereg.md` v1 (9f29867)
- `prereg_v2.md` (e873b1e)
- `prereg_v2_1.md` (9329ba2)
- `prereg_v2_1_amend1.md` (1b3ae7a)

**Status: frozen.** This file, committed as `prereg_v2_2.md` and tagged `prereg-v2.2`, is binding. Its draft history
is `prereg_v2_2_DRAFT.md` (revisions 1–8, git history up to `da72c80`).
- The commit that sets `io_wildppg.CONFIRMATORY_PPG_OPEN = True` must come after that commit.
- The confirmatory analysis is run **once**, and no method is changed after opening.
- No WildPPG confirmatory PPG has been read.
- Inputs: DaLiA (all 15, train/dev) and WildPPG dev IDs e61/u7y/ngh only.

## 0. Framing (PI 22-11)

**A pre-registered audit.** The audit asks four questions about one family of rules:
- Does a per-person, tolerance-type false-alarm guarantee hold under free-living wear?
- If it is violated, where does the violation come from?
- Can it be repaired?
- What does the repair cost?

The rules are: per-state block thresholds on wrist-PPG HR, with an SQI gate. The paper does **not** claim:
- a new method;
- a lower FAR than any baseline.

Target venue tier: Q2.

**Scope of every conclusion (L1, Cowork r2).**
- All 16 WildPPG participants followed the same one-day route, Zurich → Jungfraujoch (up to 3,571 m). Any change over
  the day is therefore likely shared by all persons, and may be unusually strong.
- The 13 confirmatory persons are not independent in the direction of such a change. The person-cluster bootstrap does
  not capture this common component.
- Every conclusion is limited to **"one in-the-wild dataset with a large altitude change"**, not to free-living wear
  in general.
- **L1 scope sentence (fixed now; it follows every conclusion sentence in §4):** "All participants of this dataset
  followed the same one-day route with an altitude change of about 3,000 m, so the persons are not independent in the
  direction of any change over the day, and the result may not carry over to free-living wear without such a change."
- An external check on a free-living dataset without a mountain route would be needed for generality
  `[Cần tìm dataset PPG cổ tay dài giờ có HR tham chiếu, license mở]`.

The outcomes are reported in a fixed order:
1. **K is the primary analysis** (H-K), followed in a fixed sequence by H-K1 (K with m = 1, §4.1b).
2. **R is a pre-registered secondary analysis.** R is interpreted as confirmatory evidence only after, and below,
   the K result.
3. R-sel is exploratory.

**No outcome of any hypothesis below is called a failure of the study.**

## 1. Frozen estimator and gate

| Item | Frozen value | Evidence (dev only) |
|---|---|---|
| E2 network | SpecFCN (amendment 1), trained on all 15 DaLiA subjects, seeds 0–4, **no fine-tuning** | `results/fcn_all_seed{0..4}.pt`. Fine-tune gain was 0.57 bpm, below the 1.0 bpm rule fixed in `06ab627` |
| E2 tracker | causal HMM forward filter, σ = 3 bpm, floor 1e-3 | DaLiA dev LOO picks 2–3 (`sigma_grid_ext.json`). WildPPG dev σ 3/5/8 changes little (`s2_analysis.json` m7) |
| Resampling | WildPPG wrist ppg_g 128 → 64 Hz and ACC 128 → 32 Hz, polyphase (`features.to_dalia_rates`) | — |
| SQI gate | a window is monitored iff the max softmax prob of SpecFCN is ≥ θ = 0.02896 | `sqi_dev.json`: smallest q passing the gated injection test; rule fixed in `b4242f7` |
| Motion state | v2 §3: rest < 0.02 g ≤ light < 0.10 g ≤ vigorous, from the std of \|acc\| per 8 s window (ACC only) | unchanged |
| Seeds | E2 seeds 0–4. **The seed is not a unit of replication** (§4.0) | — |

## 2. Rules under audit

All rules share one machinery (`04_code/h32/method.py`; R1.md §1):
- **State-time blocks.** Per state, a clock counts monitored windows of that state. Block j of state k is the j-th
  10 minutes on that clock. A block may consist of several real-time segments.
- **Debounce.** D_i = min of 3 real-time-consecutive monitored windows in the same state-time block.
- **Refractory.** One alarm start per state-time block. There is no episode merging.
- **Block score.** S = max D in the block.
- **Complete blocks only.** Only complete blocks are evaluated, so monitored test hours = number of test blocks / 6.

By Lemma 1 (R1.md §2), the number of alarms = #{test blocks with S > τ}, for every rule below.

**K (primary).** Per state:
- τ_k = the maximum of the first 6 complete blocks of that state (n_cal = 6, k* = 6). No buffer (m = 0 by the rule
  in `dev_m_estimate.py`).
- Test = complete blocks 6, 7, … of that state.
- Guarantee (R1.md Prop. 2a, under A(0)): with probability ≥ 0.912 over calibration, the long-run false-alarm rate
  of the cell is ≤ 2 per monitored hour.
- K with m = 1 (tested as H-K1, §4.1b): thinned calibration from n_cal = 12, τ = max of blocks 0, 2, …, 10 (0-based),
  buffer 1, test blocks ≥ 13.

**R (secondary).** For test block j ≥ 7 of state k:
- τ_{k,j} = max(S_{k,j−7}, …, S_{k,j−2}). That is the 6 most recent complete blocks, with a gap of 1 block.
- Guarantee (R1.md Remark 3): E[rate] ≤ (60/10)/7 = 0.857 per monitored hour per cell.
- This needs **A_loc(0): for every test block j, the 7 variables (S_{k,j−7}, …, S_{k,j−2}, S_{k,j}) are
  exchangeable.** The gap block j−1 is excluded, and the index set is fixed by the clock, not by data.
- A_loc(0) follows from "blocks j−7..j iid (or exchangeable)". It does **not** follow from stationarity plus
  1-dependence (P3, Cowork r3; R1.md Remark 3 has the MA(1) table).
- There is no PAC statement for R.

**R-thin (pre-registered sensitivity analysis for R, P3).** For test block j ≥ 13 of state k:
- τ_{k,j} = max(S_{k,j−13}, S_{k,j−11}, …, S_{k,j−3}): 6 blocks at odd lags 3..13.
- Under stationary 1-dependence the 7 variables are mutually independent and identically distributed, so E[rate] ≤
  0.857/h holds.
- Cost: 13 complete blocks (130 monitored minutes of the state) before the first test block. Cells with fewer blocks
  are not tested under R-thin.

**R-sel (exploratory, no guarantee).** As R, but the window is the 6 most recent blocks ≤ j−2 in which no alarm
started. This follows the anomaly-selective idea of Sim & Kim 2026 (10.1109/access.2026.3734990). Selecting on the
data breaks the exchangeability argument (R1.md Remark 3).

**Cold start / fallback.**
- Until state k has its first 6 complete blocks (K), 7 (R and R-sel) or 13 (R-thin), that state uses gated B0 at
  120 bpm, with the same debounce and refractory.
- Fallback alarms are reported separately and are not covered by any guarantee.
- Reported for every (person, state): the fraction of wear time with an active within-person threshold, and the hour
  at which calibration finished.
- **Q6 (Cowork r3).** The fraction of real wear time covered by a guarantee is a **main result of the "cost" part**,
  not a descriptive extra. It is reported for K, R and R-thin.
  - R needs 70 monitored minutes per state. On dev one block of 10 monitored minutes spans 23–42 real minutes, so this
    is roughly 2.7–4.9 h of real time.

## 3. Comparators

The comparators give context for sensitivity at matched FAR. They are not competitors.
- B0: one global threshold, 120 bpm.
- **B0-matched:** one global threshold, tuned on the confirmatory data to give the same pooled false-alarm count as
  K in each seed. This is an oracle that favours B0.
- **B0-state-matched:** one threshold per state, matched the same way.
- **B6 is merged with K and reported in one row "K / B6".** On dev, B6 matched to K's FAR saturates at
  q = 1.0 (the calibration maximum of D). Its per-person sensitivity is within 0.02 of K, so it is not an
  independent baseline.

All comparators see the same gated windows, debounce, refractory and test blocks.

B2′, B4, B5 and B7 (S4, Cowork r2):
- **B4 = ACI at block level** (Gibbs & Candès 2021), **exploratory**, kept because it is the standard method built
  to tolerate drift. Definition, fixed now (`05_experiments/s2/b4_aci.py`):
  - Per state, test blocks j ≥ 6 as K.
  - τ_j = the ⌈(j + 1)(1 − a_j)⌉-th smallest of all earlier complete blocks S_0..S_{j−1} (+∞ if the index > j, −∞ if
    ≤ 0).
  - a_{j+1} = a_j + γ(α* − 1{S_j > τ_j}), a_6 = α*, γ = 0.05.
  - α* = 1/7 is primary (the same expectation target as R, 0.857/h); α* = 1/3 is secondary (r = 2/h).
  - γ and α* were fixed before any run, not tuned.
- **What ACI's guarantee is.** Its guarantee is on the long-run **fraction of blocks with an alarm, true events
  included**, and it holds without exchangeability. Its FAR on clean data is near α* by construction. It is not a
  guarantee of the same kind as K's per-cell tolerance statement, and is never presented as one.
  - ACI also reacts to true events by raising its threshold (masking, §5).
  - **Finite-sample bound (D4).** Gibbs & Candès 2021, Proposition 4.1 (p. 6; full text checked 22-11, SHA256 in
    `01_lit/fulltext_notes.md` §6.7): with probability one, for all T, |(1/T) Σ err_t − α*| ≤ (max{a_1, 1 − a_1} + γ)/(γT).
    - Our block ACI meets the premise of their Lemma 4.1.
    - With γ = 0.05, a_1 = 1/7 and T = 4–33 test blocks per dev cell, the bound is 0.56–4.8. It is **vacuous in 7 of
      9 dev cells**, and in the other 2 it only gives FAR ≤ 4.2–5.1/h (`r5_aci_bound.json`).
    - **ACI's FAR is reported as an observation, not as a check of a guarantee.** γ and α* stay as fixed; no other
      value is added.
- **B5 = R-sel at block level.** R-sel is the anomaly-selective idea of Sim & Kim 2026 implemented inside the same
  machinery. There is no separate B5 row.
- **Dropped from the confirmatory set:**
  - B2′ (population split conformal calibrated on DaLiA): confounded by the device difference between datasets.
  - B7 (W1-ACAS): needs a time-series foundation model outside the frozen pipeline.
  - Neither was implemented, so there are no dev numbers for an appendix; this is stated.

## 4. Confirmatory analysis (13 WildPPG IDs, run once)

### 4.0 Units, seeds, intervals (PI item 4)
- **Units.** Persons (n = 13) and (person, state) cells. A cell needs ≥ 1 complete test block under K.
- **Seeds.** Every quantity is computed per E2 seed and then **averaged over the 5 seeds**. Seeds are never added to
  n.
- **Intervals.** By default, every interval is a **person-cluster bootstrap** over the 13 persons: 10,000 draws, seed
  0, percentile. Exact Poisson intervals are not used for inference. Exceptions, fixed here:
  - H-R and R-thin (§4.3) use the two-sided 90 % **t** interval of the mean of person rates.
  - The covariate models (§4.2 d) use a person-cluster **pairs** bootstrap: persons resampled with all their blocks,
    2,000 draws, percentile 90 % CI.
  - p_order (§4.2 a) is a permutation p-value, not an interval.
- **Per-person reporting.** Every table also reports each person separately, including persons for whom K does worse
  than a comparator.
- **Wording.**
  - Never write "at most 2 false alarms per hour".
  - Allowed: "with probability at least 0.912 over calibration, the long-run false-alarm rate of this cell is at most
    2 per monitored hour, under A(0)".
  - "Monitored hour" is state-clock time.
- **Missing persons and cells (rev 8, PI 23-11).**
  - H-K: a person without any eligible cell stays in the bootstrap with d_i = 0 (n = 13).
  - H-K1, H-R, R-thin and R − R-thin: a person is excluded from that test only if it has **no test block of that rule
    in all 5 seeds**. The actual n is reported, and every sentence "at n = …" is filled with the actual n.
  - The number of complete blocks depends on the seed, because the SQI gate uses the E2 output (dev: u7y vigorous
    9–10 blocks, ngh vigorous 15–16). Two kinds of quantity are therefore treated differently:
    - **Sum-type** (number of cells above r; d_i of X2, Y2 and H-K1): a seed without test blocks contributes 0, and
      the sum is divided by 5. Under H0 the expectation is still 0.
    - **Rate-type** (person rates of K, R and R-thin; the paired R − R-thin difference; sensitivity): an undefined
      rate is never set to 0. The rate is 6 × (mean N) / (mean T) over the seeds with ≥ 1 test block of that rule,
      and the number of seeds used is reported. (Setting it to 0 would pull the rate down and make H-R harder to
      reject.)
    - Paired R − R-thin: only the seeds in which both rules have test blocks on the common set (blocks ≥ 13).
    - When all 5 seeds have blocks (as on dev), both treatments give the same number.
  - There is no minimum number of blocks per person. Instead, next to the unweighted mean of person rates (the test),
    the pooled rate per monitored hour and each person's number of test blocks (and seeds used) are always reported,
    so that a rate resting on very few blocks is visible.
  - A raw file that cannot be opened, or lacks a required variable (wrist `ppg_g`, `acc_x/y/z`, or the HF reference),
    is read a second time. If it fails again, the failure is logged, the person is excluded from every analysis, a
    deviation is recorded automatically, and the run goes on. No partial data of that person are opened or looked
    at, and the person is not replaced by a dev person.
    - A death of the process is not "unreadable": it is handled by the rerun rule of §4.7.
    - A readable file without any monitored window is not "unreadable" either: it is handled by the rules above.
  - Descriptive tables (FAR per person, sensitivity, the altitude models, p_order, ACI bound): means over the persons
    with data, the actual n reported in each table. p_order drops a cell without test blocks from both X_K and
    X_oracle alike. The covariate pairs bootstrap resamples the persons with test blocks in that seed.
  - No person or cell is excluded for any other reason (there is no outlier rule).
  - Technical guards: if fewer than 2 persons remain for H-R or for R − R-thin, or none for H-K1, the script reports
    "not computable (n = …)" instead of a test. This happens only if at most one person (H-R, R − R-thin) or no
    person (H-K1) has test blocks of that rule.

### 4.1 H-K (primary, directional)

**H-K:** under K, the number of (person, state) cells whose observed rate exceeds r = 2/h is **larger** than
the guarantee allows.

Definitions:
- For cell c: T_c is the number of complete test blocks, N_c the number of alarms, and R_c = 6 N_c / T_c.
- Under A(0) with continuous F_k, N_c ~ BetaBinomial(T_c, 1, 6). With ties (HR grid, S = −∞) N_c is stochastically
  smaller (R1.md Prop. 2a, P1). So P0_c is an **upper bound**: X2 stays valid but loses power when ties are frequent.
  Dev tie rates: §5.
- P0_c = P0(R_c > r), computed from the realised T_c.

Test:
- Per person i: d_i = mean over seeds of Σ_{c ∈ i} (1{R_c > r} − P0_c).
- Statistic: X̄ = Σ_i d_i, the seed-averaged excess number of cells above r.
- **Decision (X2):** reject "guarantee holds" in favour of H-K iff the 5 % quantile of the person-cluster
  bootstrap of Σ_i d_i is > 0 (one-sided, α = 0.05).
- Reported alongside:
  - X1: exact Poisson-binomial tail of X per seed given T_c, cells independent; median p over seeds.
  - Y2: the same bootstrap for N_c − T_c/7, i.e. d^Y_i = mean over seeds of Σ_{c ∈ i}(N_c − T_c/7). This is the
    expectation level of K under A(0): E[N_c] = T_c/7, a mean rate of 0.857 per monitored hour. Y2 enters the
    "not rejected" sentence below (rev 7).
  - Mean rate = the unweighted mean over persons of the person rate (6 × Σ N / Σ T, seed-averaged), as for H-R.
  - If X1 or Y2 disagree with X2, this is stated, and X2 decides.
- The test is conditional on the realised T_c. T_c depends on the SQI gate, which uses PPG.

**Prediction (based on dev only): H-K is supported (rejection).**
- Dev, 3 persons: X = 3 of 9 cells in 4/5 seeds, against E0 = 1.04 (exact X1 p = 0.076).
- Cells above r: u7y light 2.17/h, ngh light 2.48/h, ngh vigorous 4.13/h (seed-averaged).
- The prediction is written before opening and is not a hypothesis about the dev data.
- **Q1 (Cowork r3).** On dev the decision statistic itself does **not** reject: X2 bootstrap 5 % quantile = −0.22,
  and X1 p = 0.076. The prediction rests on point estimates only, not on a dev rejection.

**Conclusion sentences (fixed now; each is followed by the L1 scope sentence of §0):**
- **Rejected:** "Under free-living wear (WildPPG, 13 held-out persons), x̄ of n_c (person, state) cells exceeded 2
  false alarms per monitored hour (seed average), more than the e0 expected if the per-cell tolerance guarantee held
  (person-cluster bootstrap lower bound > 0). The guarantee of Proposition 2a did not transfer to these data. Its
  assumption A(0) is violated in at least part of the cells. §4.2 states which source of the violation the data
  support."
- **Not rejected, and Y2 not rejected (rev 7):** "Under free-living wear, x̄ of n_c cells exceeded 2 false alarms per
  monitored hour, consistent with the e0 expected under the per-cell tolerance guarantee (bootstrap lower bound ≤ 0).
  The mean rate was x per monitored hour against 0.857 expected under A(0) (Y2: bootstrap 5 % quantile q). This was
  against the dev-based prediction. It does not prove the guarantee: power was about 0.75 when 20 % of cells run at
  1.5 r (simulation, §4.6, not adjusted for a drift shared by all persons)."
- **Not rejected, but Y2 rejected (rev 7):** "Under free-living wear, x̄ of n_c cells exceeded 2 false alarms per
  monitored hour, against the e0 expected under the per-cell tolerance guarantee (bootstrap lower bound ≤ 0). The mean
  rate was x per monitored hour against 0.857 expected under A(0) (Y2: bootstrap 5 % quantile q > 0). The tolerance
  statement was not refuted, but the mean rate exceeded its expectation under A(0). This was against the dev-based
  prediction. It does not prove the guarantee: power was about 0.75 when 20 % of cells run at 1.5 r (simulation, §4.6,
  not adjusted for a drift shared by all persons)." In this branch the words "consistent with the guarantee" are not
  used.
- In both "not rejected" branches the H-K1 statistic is reported as description only (fixed sequence, §4.1b).
- **Why Y2 is reported but X2 stays the primary test (rev 7).** X2 can be less sensitive than an expectation-level
  statistic, and the PI noted this on dev: X2 did not reject (quantile −0.22), while the expectation-level H-K1
  statistic (K with m = 1) was 2.4 > 0.
  - Correction recorded by the dry run (§4.7): for K with m = 0, Y2 on dev is **not** above 0 (Σd^Y = 13.3, bootstrap
    5 % quantile −1.91). The 2.4 belongs to H-K1 (m = 1).
  - X2 stays the primary test because it matches the tolerance statement of Prop. 2a (cells above r), which is what
    the guarantee says. Y2 matches only its expectation.

### 4.1b H-K1 (second test of a fixed sequence; rev 6)

**Sequence.** H-K1 is tested **only if H-K is rejected**. Each test is at 5 % (fixed-sequence procedure, no split of
α). If H-K is not rejected, H-K1 is not tested, and its statistic is reported as description only.

**Rule.** K with m = 1, as in §2:
- τ_c = the maximum of blocks 0, 2, …, 10 (thinned calibration from n_cal = 12 blocks), buffer 1, test blocks ≥ 13;
- only cells with ≥ 14 complete blocks enter.

**Statistic.** T_c = the number of test blocks (≥ 13) of cell c, and N_c = its number of alarms (= exceedances, Lemma 1).
- d_i = mean over E2 seeds of Σ_{c ∈ i} (N_c − T_c/7).
- **Reject iff the 5 % quantile of the person-cluster bootstrap of Σ_i d_i is > 0** (10,000 draws, seed 0).

**Why this statistic.** Under stationary 1-dependence, the 7 variables (the 6 thinned calibration blocks and a test
block) are independent and identically distributed, so E[N_c] ≤ T_c/7 still holds. N_c is no longer BetaBinomial,
because the test blocks are dependent. **X2 with BetaBinomial P0 is therefore not used as a test for m = 1.**

**Size and power** (`05_experiments/s2/r6_hk1.py`). 13 persons, dev block profiles, Gaussian blocks, bootstrap 500,
5 seeds × 1,000 replicates.

| Block process | P(H-K1 rejects) | P(H-K rejects) | P(both reject, sequence) |
|---|---|---|---|
| iid | **0.051** | 0.042 | 0.004 |
| MA(1), lag-1 = 0.17 | **0.046** | 0.096 | 0.006 |
| MA(1), lag-1 = 0.30 | **0.045** | 0.166 | 0.014 |
| drift 0.14 SD/block, all cells | 1.00 | 1.00 | 1.00 |
| drift 0.14 SD/block, half the cells | 1.00 | 1.00 | 0.99 |
| drift 0.07 SD/block, all cells | 1.00 | 1.00 | 1.00 |

- PI stop rule: stop if the size under MA(1) with lag-1 = 0.30 exceeds 0.07. It is 0.045, so no stop.
- H-K1 keeps its level under stationary serial dependence, whereas H-K does not (0.096–0.166).

**Conclusion sentences (fixed now; each is followed by the L1 scope sentence):**
- **H-K rejected, H-K1 not rejected:** "the excess under K is compatible with short-range dependence between
  neighbouring blocks; with thinned calibration the expectation level was not refuted at n = 13." The paper does not
  write "the guarantee holds".
- **H-K rejected, H-K1 rejected:** "the excess remains when lag-1 dependence is allowed."

### 4.2 Attribution controls (PI item 2)

**A claim that the confirmatory block scores drift, or are non-stationary, may be made only under the covariate
condition of the table below (third row).** Elsewhere these words appear only:
- as named alternatives ("serial dependence and/or drift", "as … would predict");
- in descriptions of simulations.

**(a) K-oracle** (a control only; it cannot be deployed because it needs the whole day in advance).
- Per cell and seed: 6 calibration blocks are drawn at random among all complete blocks of the cell, and the other
  blocks are tested. 2,000 draws.
- For distinct block maxima the random split is exchangeable by construction. **K-oracle therefore keeps the
  guarantee by construction** (up to ties). Its role is the reference distribution given the observed block
  values.
- Statistic: p_order = (1 + #{b : X̄_oracle,b ≥ X̄_K}) / (B + 1), B = 2,000, where X̄ is the number of cells above r
  averaged over seeds.
  - The draws are independent across cells.
  - **Draw b uses the same random keys in every seed** (common random numbers), and X̄_oracle is averaged per draw
    index.
- **Assumption of this permutation test (Q3).** Under H0, the joint law of all block scores of a person is invariant
  to independent permutations of the block order within each cell. The cells of one person are interleaved on the
  real-time axis, so this is a joint assumption, not one per cell.
- p_order < 0.05 means: the block scores are **not exchangeable in their time order**. The violation comes from the
  order of the blocks, not from which values occur.
- **p_order has power against stationary serial dependence as well as against drift (rev 5, PI item 1).** Six
  consecutive positively dependent calibration blocks have a smaller maximum than six blocks drawn at random.
  - Simulation, 13 persons, stationary MA(1): size of p_order is **0.085 at lag-1 = 0.17** (the dev pooled detrended
    value, 0.17 [0.05, 0.29]) and **0.159 at lag-1 = 0.30** (§4.6).
  - So p_order < 0.05 alone allows only: **"block scores are not exchangeable in time order (serial dependence
    and/or drift)"**.
- **p_order is the primary attribution test (S1, Cowork r2).** It tests the exchangeability that the guarantee needs,
  directly, and does not use the threshold r in its logic. Its power with 13 persons is in §4.6.

**(b) K-intact is not feasible.** Strictly: only blocks that are one real-time segment.
- On dev, 0 of 203 complete blocks is a single real-time segment.
- No real-time run of one state reaches 10 minutes. The median run is 2–5 windows.
- Descriptive substitute only (see (c)):
  - fragmentation drift: ρ_seg,c = Spearman(number of real-time segments, block index), and ρ_valid,c = the same for
    the number of finite-D windows;
  - partial drift index: ρ̃_c = the partial Spearman of S on block index given both fragmentation covariates (rank
    residuals).
- Stationary fragmentation does not break A(0); only fragmentation that changes with time does (R1.md §2, Scope).

**(c) Drift index — descriptive only (S1e).** ρ_c = Spearman(S_{c,j}, j) over all complete blocks of the cell,
seed-averaged. The cell-level association Spearman(ρ_c, R_c) and ρ̃_c are reported, with person-cluster CIs, as
**description only, never as a condition for any wording**.
- ρ_c and R_c share the test blocks, so the association is partly circular.
- Fragmentation lies on the causal path from time to block max (a changing day → more state switches → more
  fragmented blocks). ρ̃_c adjusts for a mediator and so understates drift.

**(d) Exogenous covariates (S2, Cowork r2; rev 5).** Altitude and wrist temperature from the authors' file
`WildPPG.mat`, variables `data_altitude_values` and `data_temp_wrist`.
- They are recorded independently of PPG, one value per 8 s; 0 is treated as missing.
  - Altitude is barometric.
  - **Temperature is the device-internal sensor of the wrist unit.** The file's description reads "temperature within
    device". It reflects both skin and surroundings; it is neither a skin nor an ambient measurement.
- **These channels are not PPG and are read before opening the confirmatory PPG.** On 2026-11-22 only their length and
  missing fraction were inspected for the confirmatory IDs: present for all 16 IDs, no NaN; value 0 in 5.8 % (k2s) and
  3.4 % (l38) of samples, else 0 %. Their values for confirmatory IDs were not looked at.
- Per K test block:
  - dAlt = |mean altitude of the block − mean altitude of the 6 calibration blocks| (km);
  - dTemp: the same for temperature (°C);
  - dTime = mean real time of the block − mean real time of the 6 calibration blocks (h).
- Model: block-level logistic regression, stacked over cells and persons, per E2 seed. Coefficients are averaged over
  seeds.
  - **Primary: exceedance ~ dAlt**, alone. This is the only covariate model that enters the wording.
  - Sensitivity 1: exceedance ~ dAlt + dTemp. The dTemp coefficient is reported whatever its sign.
  - Sensitivity 2: exceedance ~ dAlt + dTime. dAlt is reported with and without dTime, together with corr(dAlt, dTime).
    On a shared route, altitude and time since calibration overlap.
  - Descriptive: dTemp alone, dTime alone.
- **Intervals.** Person-cluster pairs bootstrap, 2,000 draws, percentile 90 % CI, fixed now for the 13 confirmatory
  persons. On dev (3 clusters) these intervals are descriptive only.
- **Failed fits in a bootstrap draw (rev 6).** A draw is dropped if any of these holds:
  - the outcome has no variation;
  - the optimiser fails;
  - the fit separates, defined as |coefficient| > 20.

  The number of dropped draws is reported for each model. If more than 5 % of the draws of a model are dropped, that
  model is reported only by the exceedance rate per dAlt tercile, with no coefficient or CI. If this happens to the
  primary model, the attribution table uses its first two rows only.
  - Dev (5 × 300 draws): 0 dropped for dAlt alone, 1.6 % for dAlt + dTime, 1.0 % for dTime alone.
- **This is an analysis choice drawn from dev**, not a deviation, because the prereg is not frozen. Primary = dAlt
  alone was fixed in rev 4 after seeing the dev joint model, in which dTemp had the unexpected sign. dTime was added
  in rev 5 at the PI's request. No confirmatory data were involved.
- One-sided expectation: a positive dAlt coefficient.
- **Wording:** "associated with altitude change". The paper never states a causal relation or a mechanism (for
  example hypoxia).

**Attribution rule (fixed now).** Applies only if H-K is rejected:

The wording is based only on p_order and the **primary dAlt model**. ρ, ρ̃, the association, dTemp and dTime are never
conditions.

| Condition | Allowed wording (followed by the L1 scope sentence) |
|---|---|
| p_order ≥ 0.05 | "the data do not show that the time order of the blocks is the source of the violation" |
| p_order < 0.05, and the 90 % CI of the dAlt coefficient (primary model) does not lie above 0 | "block scores are not exchangeable in time order (serial dependence and/or drift); exceedances were not associated with altitude change" |
| p_order < 0.05, and the 90 % CI of the dAlt coefficient (primary model) lies above 0 | "block scores are not exchangeable in time order, and exceedances were associated with altitude change since calibration (non-stationarity over the day); physiological and behavioural components were not separated" |

In the third row, if the dAlt CI of sensitivity model 2 (with dTime) contains 0, this is added: "altitude change and
time elapsed since calibration could not be separated."

Only the third row asserts drift or non-stationarity. The second row names drift only as one of two possible sources.
Neither row makes a physiological claim.

### 4.3 R (secondary, interpreted only after §4.1–4.2 are reported)

**H-R:** the mean R rate exceeds the expectation bound 0.857/h.

**Estimand (fixed now, Q2):** the unweighted mean over the 13 persons of the person rate.
- Person rate = 6 × Σ_states N / Σ_states T over that person's R test blocks, seed-averaged.
- The bound holds for this estimand and for the pooled per-hour rate. Only the unweighted mean is tested.

Tests:
- **Primary:** one-sample t test, one-sided, H0: mean ≤ 0.857 vs H1: mean > 0.857, at α = 0.05. Reported with a
  two-sided 90 % t CI, which is equivalent to the one-sided test.
- **Size rule (fixed now):** if the simulated size of this t test under iid blocks (§4.6) exceeds 0.06, the primary
  test is the bootstrap-t (studentised person-cluster bootstrap, 9,999 draws) instead. Simulated size: 0.041 (§4.6),
  so **the t test is the primary test**.
- **Sensitivity:** Wilcoxon signed-rank (one-sided) and the person-cluster percentile bootstrap. Wilcoxon tests a
  pseudo-median, not the mean, so it is not the primary test.
- The same is reported per person, with no test.
- **R-thin:** the same estimand and tests, reported next to R as a pre-registered sensitivity analysis.

Conclusion sentences:
- **H-R not rejected:** "Rolling calibration kept the long-run rate consistent with its expectation bound of
  0.857/h (mean of persons x, 90 % CI …): **not refuted at n = 13**, which is not the same as "the bound holds".
  This came at a sensitivity cost of … and with masking of … after an event. Rolling calibration gives only an
  expectation bound, not the per-cell tolerance guarantee."
- **H-R rejected:** "Rolling calibration did not keep the expectation bound: the 7 blocks of a window were not
  exchangeable. This does **not** by itself indicate drift. Stationary short-range dependence between neighbouring
  blocks (lag 1) is enough to raise the exceedance probability above 1/7 (R1.md Remark 3: MA(1) gives 1.04–1.10/h).
  The R-thin result is reported next to it."

Prediction from dev (rewritten in rev 5 for the t test): **uncertain**.
- Dev (3 persons): mean of person rates 1.02/h, 90 % t CI [0.63, 1.42], one-sided t p = 0.17, so **not rejected**.
- Per person: e61 1.05, u7y 0.77, ngh 1.24.
- The point estimate is above the bound, but no direction is predicted.

**R vs R-thin, paired (descriptive, not a test; PI B1–B3).** R is also computed on exactly the R-thin test blocks
(blocks ≥ 13 of the same person and state), and the difference R − R-thin is reported per person and as the mean
over persons, with a 90 % t CI.

The direction is predicted in advance:
- Short-range dependence between neighbouring blocks predicts R − R-thin > 0, because R's window is contiguous and
  R-thin's is spread out. Simulation: MA(1) gives +0.17 (θ = 0.5) and +0.24 (θ = 1.0).
- Upward drift predicts R − R-thin ≤ 0, because R-thin's 13-block window reaches further back to lower values.
  Simulation: drift gives −0.34 to −0.69.

Wording, fixed now:
- **R − R-thin > 0, and its 90 % CI lies above 0:** "on the same test blocks, the contiguous rolling window gave more
  false alarms than the spread window, as short-range dependence between neighbouring blocks predicts; this does not
  exclude drift."
- **R − R-thin ≤ 0, and its 90 % CI lies below 0:** "on the same test blocks, the spread window gave at least as many
  false alarms, as an upward change over the day would predict and short-range dependence alone would not."
- **CI contains 0:** "the paired comparison does not separate short-range dependence from non-stationarity."

Lag-1 and lag-2 autocorrelation of block scores are reported per person and pooled, raw and after a linear detrend,
with 90 % CIs (Fisher z). Description only.

R-sel (exploratory):
- Reported: FAR and the same bootstrap; sensitivity.
- **Mechanism (L2, Cowork r2), stated with the result.** Dropping blocks with an alarm from the window truncates the
  upper tail of the calibration blocks. The threshold is pulled down, and FAR rises mechanically; it is not a chance
  finding.
- **Masking after an event**, by block-level simulation as in `05_experiments/s2/controls.py`:
  - a first event's score is inserted in test block j0;
  - detection of a second event is measured h = 1..8 state-blocks later;
  - reported as detection with vs without the first event, for K, R, R-thin, R-sel and B4 (ACI).

### 4.4 Descriptive and secondary
- **FAR_any per cell, person and pooled.** Reported with monitored hours, test blocks, coverage and n = 13.
- **Sensitivity and delay on injected episodes.** Δ 20/30/40, ramp 30 s, plateau 180 s; split by state and by
  person. Gated-out injected events count as missed.
  - Threshold estimated on the clean stream (main).
  - Threshold estimated on the injected stream itself (self). Injection is dense (3/h), so this is a worst case for
    contamination.
- **Matched-FAR comparison.** K vs B0-matched and B0-state-matched: sensitivity difference per person and averaged
  over persons, with a person-cluster CI.
  - **Q4 (Cowork r3).** B0-matched and B0-state-matched are tuned on the confirmatory data. They are oracles that
    favour B0 and cannot be deployed. The difference K − B0-matched is therefore a **lower bound** for K.
- **Limitation (Cowork m8).** Injected events are a semi-synthetic time-warp of real PPG (an HR ramp and plateau).
  They are not arrhythmias and are not called so.
- **Calibration contamination.** As in `s2_analysis.json`: one +30 bpm event inserted in calibration block 2 of each
  state.

### 4.5 Order of reporting

The results are reported in this order:
1. §4.1 H-K.
2. **§4.1b H-K1** (K with m = 1), immediately after H-K: per-cell rates, d_i, the bootstrap decision, and the K
   (m = 0) rates on the same test blocks (≥ 13) for a paired view.
3. §4.2 attribution controls.
4. §4.3 R, then R-thin, then R-sel.
5. §4.4 descriptive and secondary results.

R results are not mentioned in the abstract unless the K result appears before them.

### 4.6 Power of H-K (simulation, `05_experiments/s2/m3_power.py`)

Simulation set-up:
- T_c is drawn from the 3 dev persons' test-block profiles [rest, light, vigorous] = [20, 21, 10], [11, 26, 4],
  [14, 33, 10].
- 13 persons; 5 seeds × 2,000 replicates; bootstrap 500.

| bad cells f | p in bad cells | X2 | X1 | Y2 | Y1 | mean X |
|---|---|---|---|---|---|---|
| 0 (H0) | Beta(1,6) | 0.04 | 0.03 | 0.05 | 0.05 | 4.6 |
| 0.2 | 1.25 r | 0.61 | 0.60 | 0.57 | 0.61 | 9.3 |
| 0.2 | 1.5 r | 0.75 | 0.76 | 0.69 | 0.76 | 10.5 |
| 0.2 | 2 r | 0.84 | 0.83 | 0.82 | 0.91 | 11.3 |
| 0.4 | 1.25 r | 0.96 | 0.97 | 0.96 | 0.97 | 14.0 |
| 0.4 | 1.5 r | 0.99 | 1.00 | 0.99 | 0.99 | 16.4 |
| 0.4 | 2 r | 1.00 | 1.00 | 1.00 | 1.00 | 18.0 |

Limits of the simulation:
- It uses one seed per replicate, not a seed average.
- p is fixed in bad cells, whereas drift gives a time-varying p.
- Cells are independent.
- The power of H-R was not simulated.

**Power of p_order, the primary attribution test (S1, `05_experiments/s2/s1e_attrib.py`).**

Simulation set-up:
- Each synthetic person = one dev person's 3 cells, as complete-block sequences of a random E2 seed.
- Each sequence is randomly permuted, which gives exchangeable blocks.
- Alternatives add a linear drift β·j (bpm per block) to a fraction f of the cells, or to all cells (common drift,
  as in L1). β = 2.8 bpm/block is the ngh vigorous slope (2.1–3.5 over seeds).
- B = 999; 13 persons; 5 seeds × 1,000 replicates.
- One sequence per cell, so there is no seed averaging; real data average 5 seeds, so power may differ slightly.
- X2 (H-K) is computed on the same data for comparison.

| Alternative | p_order | X2 | mean X_K |
|---|---|---|---|
| H0 (permuted) | **0.034** | 0.022 | 4.0 |
| f = 0.2, β = 1.4 | 0.34 | 0.27 | 7.0 |
| f = 0.2, β = 2.8 | 0.65 | 0.56 | 8.9 |
| f = 0.4, β = 1.4 | 0.77 | 0.73 | 10.0 |
| f = 0.4, β = 2.8 | 0.98 | 0.97 | 13.9 |
| all cells, β = 1.4 | 1.00 | 1.00 | 19.0 |
| all cells, β = 2.8 | 1.00 | 1.00 | 28.6 |

- p_order keeps its level (0.034) and is slightly more powerful than X2.
- These H0 rows permute the blocks, so they also remove serial dependence. They do not show how p_order behaves under
  stationary serial dependence; the next table does.

**Size of p_order and X2 under stationary serial dependence (rev 5, `05_experiments/s2/r5_porder_ma1.py`).**
Gaussian MA(1) blocks with the given lag-1 autocorrelation, no drift; dev block profiles; 13 persons; B = 999;
5 seeds × 1,000 replicates.

| lag-1 | p_order | X2 (K, m = 0) | Mean rate K (m = 0) | Mean rate K (m = 1) |
|---|---|---|---|---|
| 0 (iid) | 0.034 | 0.040 | 0.858 | 0.858 |
| 0.17 (dev pooled) | **0.085** | 0.101 | 0.921 | 0.859 |
| 0.30 (dev ngh) | **0.159** | 0.174 | 0.977 | 0.858 |

- Pre-set stop rule (PI): stop if the size at lag-1 = 0.17 exceeds 0.10. It is 0.085, so no stop.
- p_order and X2 both reject more often under stationary dependence; K with m = 1 keeps the expectation of 0.857/h.
- This is why p_order alone does not allow a claim of drift or non-stationarity (§4.2).
- Power is low (0.34) when only 20 % of cells drift at half the dev slope.
- Power is ≈ 1 under a common drift, which is the L1 scenario.

**Size of the H-R tests under iid blocks** (`05_experiments/s2/r3_checks.py`).
- Set-up: 13 persons; block profiles drawn from the 3 dev persons (complete blocks per state, seed 0: e61 26/27/16,
  u7y 17/32/10, ngh 20/39/16); iid continuous block scores, so E[rate] = 0.857 exactly (the boundary of H0).
- Runs: 5 seeds × 20,000 replicates; bootstrap-t on 5 × 2,000 replicates.
- Nominal level: 0.05.

| Rule | t (primary) | Wilcoxon | percentile bootstrap | bootstrap-t |
|---|---|---|---|---|
| R | 0.041 (sd 0.002) | 0.045 | **0.062** | 0.046 |
| R-thin | 0.039 | 0.029 | 0.061 | 0.046 |

The t test keeps its level, so the size rule does not trigger. The percentile bootstrap is slightly anticonservative
(0.06), which is why it is only a sensitivity analysis.

### 4.7 Confirmatory script, dry run and environment (rev 7, updated in rev 8)

**Single entry point.** The confirmatory analysis is
`uv run python -W ignore 05_experiments/confirm/run_confirmatory.py --ids <13 IDs> --label confirmatory`
(Makefile target `confirm`). The dev dry run and the confirmatory run use the same code path; the only parameter is
the ID list (plus an output label).
- All analysis code is in `04_code/h32/confirm.py`, moved unchanged in logic from the dev scripts of §5. It calls
  only the verified functions of `04_code/h32/method.py`.
- All seeds are fixed in the file:
  - inference seeds 0–4;
  - person-cluster bootstrap seed 0, 10,000 draws;
  - p_order: B = 2,000, oracle keys `default_rng(1000 · person index + state)`;
  - covariate pairs bootstrap seed 11, 5 × 400 = 2,000 draws;
  - masking seed 3, association seed 20261123.
- While `io_wildppg.CONFIRMATORY_PPG_OPEN = False`, the script refuses any confirmatory ID before any file is read
  ("REFUSED: confirmatory IDs …"); test `04_code/tests/test_confirm_branches.py`.
- The missing-data rules of §4.0 are implemented and tested (`04_code/tests/test_confirm_missing.py`, synthetic
  persons built from the dev arrays):
  - a person without blocks in all 5 seeds;
  - a person with blocks in 2 of 5 seeds;
  - a person with exactly 1 test block;
  - a raw file without the `wrist` variable, without `ppg_g`, and a file that is not a MAT file (each read twice,
    then excluded);
  - the lock error is never treated as "unreadable";
  - the actual n in every sentence "at n = …".
- Every conclusion of §4.1–4.3 is filled automatically from the fixed wording tables. The script prints the branch
  taken and its reason (`runs/<label>/conclusions.md`).
  - A test on fake results reaches every branch: H-K 3 branches; H-K1 not tested / rejected / not rejected;
    attribution not applied / rows 1–3; H-R; R vs R-thin; and the "not computable" guards (§4.0).
  - The same test checks the time clause and that "consistent with the guarantee" never appears when Y2 rejects.
  - 24/24 tests pass (rev 8).

**Code and environment hashes** (the confirmatory run must reproduce these; `meta.json` of the run records them
again):

| Item | Value |
|---|---|
| Script commit | `d4ce6f2` (rev 8: confirm.py, run_confirmatory.py, tests); dry runs at HEAD `d4ce6f2` |
| SHA256 `04_code/h32/confirm.py` | `58624dde442db9cdbf23534bafef248db4d283062bc22368d4d7bed9561d1488` |
| SHA256 `05_experiments/confirm/run_confirmatory.py` | `1a49ee405adea825a38bf64954503ed88386326f8256d2d92acf67747bcfc28b` |
| SHA256 `04_code/h32/method.py` | `17e84a0a5cf22e1bb21f8c65eb2d6e08361978bb1e531f2bbdbb8dbf26a9e1c9` |
| pip freeze | `05_experiments/confirm/env_pip_freeze.txt`, SHA256 `a3d2db51773e779e8a29361726d16a2c7e84cd8b8138f51313536e3ca1d9cf0e` (numpy 2.1.3, scipy 1.14.1, torch 2.5.1) |
| SHA256 `uv.lock` | `1e5a22e87914ee37ed81446a2fdbb2d4f315835598ded4bd5d15fa6c3b938206` |
| Python / platform | 3.12.16 / macOS-26.6.2-arm64 (Apple silicon, CPU, `torch.set_num_threads(8)`) |

**Dry run on the 3 dev IDs** (e61, u7y, ngh; rev 8 labels `dryrun_dev_r8_1`, `dryrun_dev_r8_2`, made by one call of
`make confirm_dry` at `d4ce6f2`; from the raw WildPPG files, not from the dev cache). Regression check:
`05_experiments/confirm/regression_check.py` → `regression_check.md` / `.json`.
- The rev 7 dry runs (`dryrun_dev_1`, `dryrun_dev_2`, script `9b99fd5`) are kept as a record and gave the same
  numbers and the same `conclusions.md`, byte for byte.
- **Prereg numbers vs the new script.**
  - 63 numbers from §4 and §5 were checked, with a tolerance of half a unit of the last printed digit; 62 match.
  - The one mismatch is the upper limit of the dAlt 90 % CI: 1.32 in the script against 1.26 in §5.
    - Cause: Monte Carlo. The script uses the 2,000 pairs-bootstrap draws that §4.2 d prescribes (5 × 400). The dev
      script `s1e_attrib.py` used 5 × 300 = 1,500.
    - With 3 clusters, the upper percentile moves. The point estimate (+0.60) and the lower limit (0.29) are
      identical.
    - The §5 number is left as it was recorded; the script follows §4.2 d.
- **E2 arrays.** 225 of 225 arrays (3 persons × 5 seeds × 15 keys: states, SQI, HR track, injected onsets and tracks
  at +20/30/40, calibration contamination) are byte-identical to the dev cache `02_data/derived/s2_cache`.
- **Run 1 vs run 2.**
  - The arrays (225/225), `results.json` (SHA256 `94ac3a08324b9d54261dbcc8195582713e2d8693ee1ccef921730bff27e0db7e`
    in both) and `conclusions.md` are byte-identical.
  - `results.json` differs from rev 7 only by the fields added in rev 8 (n, seeds used, test blocks per person) and
    the deleted B0-state entry. Every rev 7 value is unchanged.
  - `run.log` differs only in the timing line, and `meta.json` only in its timestamps and timings.
- **Branches filled on dev:**

| Analysis | Branch | Reason |
|---|---|---|
| H-K | not rejected; Y2 not rejected | X2 quantile −0.22 ≤ 0; Y2 quantile −1.91 ≤ 0 |
| H-K1 | not tested (fixed sequence); statistic shown as description only | Σ d = 13.29, quantile 2.43 |
| Attribution | not applied | H-K not rejected |
| H-R | not rejected | t p = 0.174 |
| R vs R-thin | CI contains 0 | 90 % CI [−0.33, 0.31] |

**Run time.**
- Dev, 3 persons, rev 8: inference 789 s and 720 s; analysis 25 s and 11 s; the machine was not shared.
- Rev 7, for comparison: inference 4,569 s and 4,769 s, with another project using about 9 of 10 cores.
- Estimate for the 13 confirmatory persons: about 53–58 min of inference on an unshared machine (5.6–5.8 h under the
  rev 7 load), plus a few minutes of analysis.
  - The estimate scales the dev time by total raw-file size: 15.96 GB against 3.64 GB, ratio 4.38.
  - The file sizes were read from file-system metadata only; no confirmatory file was opened.

**B0-state (rev 8).** The rev 7 dry run found that §3 listed "B0-state: one fixed threshold per state" without
values. B0-state is deleted (deviation row 20). B0-state-matched stays.

**What "run once" means (rev 8, PI).**
- E2 inference is completed for all 13 persons before any analysis is run. No result of any part is printed or
  looked at before all 13 persons are done.
- If the process dies for a technical reason (power loss, out of memory), a rerun with exactly the hashes of this
  section is allowed.
  - The time and cause are written to the run log.
  - Arrays already completed are reused.
  - This is not an analysis deviation.
- Any change of code after opening is a deviation. Its diff and reason are recorded, and results before and after
  the change (if both exist) are both reported.
- The run uses the machine and environment of this section.

## 5. Dev evidence used for this pre-registration (exploratory; WildPPG e61/u7y/ngh; seed-averaged)

Sources: `05_experiments/s2/results/{s2_analysis,drift_extra,controls,m3_power,m6_curve}.json`. Figure:
`05_experiments/s2/figs/M2_blockmax_WildPPG.png`.

| Cell | K rate (/h) | seeds > r | oracle P(>r) | ρ | ρ̃ (partial) | ρ_seg~time | R (/h) | R-sel (/h) |
|---|---|---|---|---|---|---|---|---|
| e61 rest | 0.36 | 0/5 | 0.07 | −0.04 | −0.02 | −0.07 | 0.82 | 0.82 |
| e61 light | 0.86 | 0/5 | 0.08 | +0.06 | +0.16 | −0.22 | 1.44 | 3.78 |
| e61 vigorous | 0.00 | 0/5 | 0.11 | −0.06 | −0.09 | −0.07 | 0.67 | 0.67 |
| u7y rest | 0.00 | 0/5 | 0.10 | −0.06 | −0.01 | −0.17 | 0.12 | 0.12 |
| u7y light | 2.17 | 4/5 | 0.09 | +0.32 | −0.03 | +0.46 | 0.94 | 3.99 |
| u7y vigorous | 1.30 | 0/5 | 0.13 | +0.03 | +0.18 | +0.24 | 1.80 | 1.80 |
| ngh rest | 0.00 | 0/5 | 0.13 | −0.13 | +0.20 | −0.60 | 0.74 | 0.83 |
| ngh light | 2.48 | 4/5 | 0.08 | +0.03 | +0.25 | −0.35 | 1.34 | 4.28 |
| ngh vigorous | 4.13 | 5/5 | 0.13 | +0.62 | +0.32 | −0.61 | 1.63 | 3.40 |

Order and association:
- K-oracle:
  - p_order per seed = 0.056 / 0.070 / 0.568 / 0.052 / 0.068 (X_K = 3 / 3 / 1 / 3 / 3, against E_oracle[X] =
    0.81–1.00).
  - **Prereg statistic (seed-averaged X̄): X̄_K = 2.6 against a mean X̄_oracle of 0.92, p_order = 0.040.** Computed
    as (1 + #)/(B + 1) since rev 3.
  - Common random numbers across seeds are required. With independent draws per seed the same statistic gave
    p = 0.000, because averaging shrinks the oracle's variance.
- Dry run of the frozen statistics on dev (3 persons; not a test of anything):
  - X2: Σd = 1.57, bootstrap 5 % quantile −0.22, so not rejected.
  - Association (descriptive), person-cluster 95 % CI: ρ_c 0.86 [0.81, 1.00]; ρ̃_c 0.53 [−0.50, 1.00].
  - H-R (t test, §4.3): mean of person rates 1.02/h, 90 % t CI [0.63, 1.42], p = 0.17, so not rejected.
- Association Spearman(ρ_c, R_c) = 0.76–0.93 over seeds (cell permutation p 0.0006–0.014). This is partly circular,
  see §4.2(c).

FAR, person-cluster bootstrap over 3 persons (coarse):

| Rule | Pooled FAR | 95 % CI |
|---|---|---|
| K | 1.40/h | 0.49–2.16 |
| R | 1.05/h | 0.77–1.24 |
| R-sel | 2.71/h | 2.03–3.29 |

Sensitivity at +20 / +30 / +40 (431 / 428 / 439 events, all seeds):

| Rule | Clean-stream τ | Self τ |
|---|---|---|
| K | 0.40 / 0.55 / 0.55 | 0.19 / 0.21 / 0.24 |
| R | 0.29 / 0.45 / 0.50 | 0.11 / 0.13 / 0.16 |
| R-sel | 0.48 / 0.58 / 0.57 | 0.36 / 0.32 / 0.40 |

Masking at Δ 30: a second event 2–7 state-blocks after the first is detected with 0.25–0.27 instead of 0.40–0.46
under R; the drop is ≤ 0.02 under R-sel and 0 under K.

Matched FAR, K − B0-matched sensitivity (seed-averaged):
- per person (e61, u7y, ngh): +20: −0.20 / +0.35 / +0.18; +30: −0.11 / +0.31 / +0.15; +40: −0.05 / −0.02 / +0.03;
- mean over persons: +0.11 / +0.12 / −0.01.

Revision 3 additions (dev, `05_experiments/s2/results/r3_checks.json`):

**Ties (P1).**
- The HMM output lies on a grid of 0.8235 bpm (127 distinct values in e61).
- Test blocks with S = τ under K, by cell:
  - e61 rest 9 %, ngh light 5 %, ngh vigorous 2 %, u7y light 1.6 %, other cells 0;
  - 1.9 % pooled.
- Blocks with S = −∞: 0.

**R vs R-thin** (mean of person rates, 90 % t CI, one-sided t p):
- R: 1.02/h [0.63, 1.42], p = 0.17.
- R-thin: 0.88/h [0.46, 1.30], p = 0.44.
- Per person (e61, u7y, ngh): R 1.05 / 0.77 / 1.24; R-thin 0.68 / 0.80 / 1.16.
- The pooled R rate (1.05/h) and the mean of persons (1.02/h) are different estimands. The t test uses the mean.
- R-thin sensitivity: 0.21 / 0.33 / 0.36, against 0.29 / 0.45 / 0.50 for R. Events in blocks 7–12 are missed by R-thin
  by construction.

**Masking (Δ 30; only the 2 cells with ≥ 28 blocks, u7y light and ngh light).** Detection of a second event:
- Under R it drops at lags 2–7 (0.19–0.24 → 0.13–0.17).
- Under R-thin it drops only at odd lags 3–13 (0.21–0.35 → 0.14–0.23), so it spreads over about 2 h of the state.

Revision 4 additions (dev; `05_experiments/s2/results/{s1e_attrib,b_rchecks,b4_aci}.json`):

**R vs R-thin on the same test blocks (B1).** Blocks ≥ 13; seed-averaged; test hours 5.0 / 3.7 / 5.9 per person.

| | e61 | u7y | ngh | Mean of persons |
|---|---|---|---|---|
| R | 0.88 | 0.64 | 1.09 | 0.87 |
| R-thin | 0.68 | 0.80 | 1.16 | 0.88 |
| R − R-thin | +0.20 | −0.16 | −0.07 | −0.01 (90 % CI −0.33 to +0.31) |

- The earlier gap (R 1.02 against R-thin 0.88) came from the different test sets, not from the window.
- The prediction rule would give "does not separate short-range dependence from non-stationarity".

**Autocorrelation of block scores (B2).** Fisher z pooled with weights n − 3, 90 % CI.

| | Raw lag 1 | Raw lag 2 | Detrended lag 1 | Detrended lag 2 |
|---|---|---|---|---|
| Pooled | 0.22 [0.09, 0.33] | 0.15 [0.03, 0.27] | 0.17 [0.05, 0.29] | 0.10 [−0.02, 0.22] |

- By person, detrended lag 1: e61 0.14 [−0.08, 0.34], u7y 0.04 [−0.19, 0.27], ngh 0.30 [0.10, 0.47].
- Positive lag-1 dependence is present after detrending, carried mainly by ngh. It is small and has the size of the
  pooled dev estimate used for the "m = 0" rule (0.159).
- The CIs treat the 5 E2 seeds as one realisation and cells as independent.

**Power of the H-R t test (B3).** 13 persons, dev block profiles, Gaussian blocks, 5 seeds × 4,000 replicates.

| Model | Power, R | Power, R-thin | Mean rate R / R-thin (/h) | Paired R − R-thin |
|---|---|---|---|---|
| iid | 0.044 | 0.041 | 0.857 / 0.857 | 0.000 |
| MA(1) θ = 0.5 | 0.64 | 0.037 | 1.03 / 0.86 | +0.17 |
| MA(1) θ = 1.0 | 0.84 | 0.040 | 1.09 / 0.86 | +0.24 |
| drift 0.14 SD/block, all cells | 1.00 | 1.00 | 1.72 / 2.41 | −0.69 |
| drift 0.14 SD/block, half the cells | 0.98 | 0.99 | 1.29 / 1.64 | −0.35 |
| drift 0.07 SD/block, all cells | 1.00 | 1.00 | 1.26 / 1.60 | −0.34 |

- The drift slope comes from ngh vigorous: 2.8 bpm per block over a residual SD of 20.6 bpm. Applied to cells of up to
  39 blocks, this is a strong alternative.
- R-thin is blind to MA(1), as designed. Both rules detect the dev-sized drift.

**B4 = ACI at block level (exploratory).** Seed-averaged; FAR per person (e61, u7y, ngh). Sensitivity at +20/+30/+40.

| α* | FAR per person | Mean FAR | Sensitivity, clean τ | Sensitivity, self τ |
|---|---|---|---|---|
| 1/7 | 0.64 / 0.75 / 0.77 | 0.72 | 0.32 / 0.49 / 0.53 | 0.08 / 0.12 / 0.13 |
| 1/3 | 1.41 / 1.79 / 1.92 | 1.71 | 0.55 / 0.67 / 0.66 | 0.26 / 0.30 / 0.30 |

- Masking (Δ 30, α* = 1/7): detection of a second event drops from 0.42 to 0.27 one block later and recovers slowly
  (0.36 after 8 blocks). This is because ACI counts the event as an error and raises its threshold.
- With self-computed thresholds on the densely injected stream, ACI's sensitivity collapses (0.08–0.13).

**Exogenous covariates (S2), dev dry run.** 149 K test blocks per seed, 3 persons, cluster bootstrap with 3 clusters
(coarse). Logistic coefficients, seed-averaged, with 90 % CIs:

| Model | Coefficient | 90 % CI | Note |
|---|---|---|---|
| dAlt alone (primary) | **+0.60 per km** | [0.29, 1.26] | |
| dTemp alone (secondary) | −0.20 per °C | [−0.33, 0.06] | opposite to the expected sign |
| Joint, dAlt | +0.72 | [0.43, 1.40] | |
| Joint, dTemp | −0.38 | [−0.52, −0.10] | |

- Exceedance rate by dAlt tercile (edges 0.84 and 2.33 km): 0.12 / 0.20 / 0.38.
- corr(dAlt, dTemp) = 0.31.
- Wrist temperature also reflects skin contact and clothing, so its sign is not interpreted.

**Rev 5 covariate models (dev, `s1e_attrib.json` key `rev5`; 3 clusters, descriptive).**

| Model | Coefficient | 90 % CI |
|---|---|---|
| dAlt alone (primary) | +0.60 per km | [0.29, 1.26] |
| dAlt + dTime: dAlt | +0.76 per km | [0.53, 1.53] |
| dAlt + dTime: dTime | +0.16 per h | [0.01, 0.30] |
| dTime alone | +0.04 per h | [−0.05, 0.15] |
| dAlt + dTemp: dTemp | −0.38 per °C | [−0.52, −0.10] |

- corr(dAlt, dTime) = **−0.33**. The route goes up and comes back down, and calibration happens early at low
  altitude, so later blocks are not always higher.
- corr(dAlt, dTemp) = 0.31.
- dTime ranges from 0.7 to 13.4 h.

On dev, the attribution rule would read: "block scores are not exchangeable in time order, and exceedances were
associated with altitude change since calibration (non-stationarity over the day) …" (p_order = 0.040; dAlt CI above
0). The time clause would not be added, because dAlt stays above 0 with dTime.

**K m = 0 against K m = 1 on the same test blocks (≥ 13), dev (`r6_hk1.json`, rev 6).** 8 cells with ≥ 14 blocks;
seed-averaged rates per hour.

| Cell | Test blocks | K, m = 0 | K, m = 1 | Same τ (seeds) |
|---|---|---|---|---|
| u7y light | 18.4 | 2.67 | 2.67 | 5/5 |
| ngh light | 25.4 | 3.06 | 3.06 | 4/5 |
| ngh vigorous | 2.8 | 3.00 | 2.20 | 0/5 |
| e61 rest | 13 | 0.55 | 0.74 | 3/5 |
| e61 light | 14 | 0.43 | 0.00 | 0/5 |
| e61 vigorous | 3 | 0.00 | 2.40 | 0/5 |
| u7y rest | 4 | 0.00 | 0.60 | 0/5 |
| ngh rest | 7 | 0.00 | 0.34 | 0/5 |

- Per person (m = 0 against m = 1): e61 0.44 against 0.56; u7y 2.20 against 2.30; ngh 2.45 against 2.45.
- Mean of persons: 1.70 against 1.77.
- **In the two light cells the thinned calibration maximum equals the m = 0 maximum in most seeds.** So on dev,
  allowing lag-1 dependence leaves their excess unchanged on the same test blocks. The thinned rule does not lower it.
- H-K1 statistic on dev (description only, 3 clusters): d_i = −1.5 (e61), +5.4 (u7y), +9.4 (ngh); Σ d = 13.3, bootstrap
  5 % quantile 2.4 > 0.
  - In the fixed sequence H-K1 would not be tested, because H-K (X2) is not rejected on dev (quantile −0.22).

## 6. Deviations, consolidated from v1 (each earlier file stays byte-unchanged)

| # | Where (version) | Original | Now | Reason / authority | Commit |
|---|---|---|---|---|---|
| 1 | v1 P4, draft M | event = HR > 120 as primary | HR > 120 only as comparison; in-control/out-of-control framing with semi-synthetic injection | PI 21-11: exercise tachycardia in healthy volunteers is not alarm-worthy; S1b B | v2 `e873b1e`, v2.1 `9329ba2` |
| 2 | v1 draft M | activity names | ACC motion-intensity states | not available on device | v2 |
| 3 | v1 draft M | baselines B0–B6 | B2′ and E2 added (v2); now §3: B0, B0-matched, B0-state-matched, B6 merged with K (B0-state deleted, row 20); B4 (ACI) exploratory, B5 = R-sel, B2′/B7 dropped (row 16) | E1 too poor (v2); audit framing (PI 22-11); B6 saturates at matched FAR | v2; v2.2 |
| 4 | v1 draft M | DaLiA 12 confirmatory, WildPPG untouched | DaLiA all train/dev; WildPPG 3 dev + 13 confirmatory | per-hour within-person test needs long recordings (S1b C4) | v2, v2.1 |
| 5 | v2 §5 / v2.1 §3 | E2 = SpecCNN + HMM | SpecFCN + HMM | shortcut learning on dev | amend. 1 `1b3ae7a` |
| 6 | v2.1 | no gate; FAR per wear hour | SQI gate θ = 0.02896; FAR per monitored hour | zero-shot E2 failed injection +40 on WildPPG dev; PI 22-11 | v2.2 |
| 7 | v2.1 §3 | optional fine-tune | not used | gain 0.57 < 1.0 bpm (pre-set rule) | v2.2 |
| 8 | v2.1 §4 | σ chosen on dev | σ = 3 fixed | DaLiA dev LOO | v2.2 |
| 9 | v2.1 §4 | calibration = first 60 min of the recording; block state by majority; ≥ 50 % monitored blocks; conformal order 5/6 | per-state sequential calibration on state-time blocks; τ = max of the first 6 complete blocks (K) | 0–3 blocks per state in 60 min (dev); majority state is not causal; T4 Prop. 2a | v2.2 |
| 10 | v2.1 §4 | cold start = population B2′ (LOSO over confirmatory calibration) | gated B0 at 120 bpm, reported separately, no guarantee | simpler; uses no other person's confirmatory PPG | v2.2 |
| 11 | v1 M1 question | "does the threshold keep FA/h ≤ target … at what sensitivity cost vs B0–B6" | pre-registered audit: does the guarantee hold under free-living wear, why not, can it be repaired, at what cost; no method claim, no FAR-reduction claim | PI 22-11 after S2 r1 (dev drift evidence) | v2.2 |
| 12 | v2.1 primary | per-subject FAR ≤ r, fraction of subjects meeting target | H-K (cells above r vs guarantee, person-cluster bootstrap); attribution controls; R as hierarchical secondary | Cowork M3; PI items 1–3 | v2.2 |
| 13 | v1 M4–M5 | Poisson + subject bootstrap; mean ± sd over seeds | person-cluster bootstrap only; seed-averaged; seeds not in n | PI item 4 | v2.2 |
| 14 | v1 M3 | buffer ≥ 4 windows | buffer in blocks: K g = 0 (m = 0 rule), R g = 1, m = 1 sensitivity g = 1, R-thin odd lags 3..13 | block-level decision unit (R1.md); R-thin added after Cowork r3 P3 | v2.2 |
| 15 | rev 2 §4.3 | H-R by person-cluster percentile bootstrap of the pooled rate | one-sided t test on the unweighted mean of person rates (size 0.041); Wilcoxon and bootstrap as sensitivity | Cowork r3 Q2 (bound is on the mean; percentile bootstrap size 0.062) | v2.2 |
| 16 | rev 2 §3 | B4 proposed dropped | B4 = ACI at block level kept as exploratory (α* = 1/7 primary, 1/3 secondary, γ = 0.05); B5 = R-sel; B2′, B7 dropped with reasons | Cowork r2 S4 | v2.2 |
| 17 | rev 2 §4.2 | attribution by p_order + CIs of ρ, ρ̃ | p_order primary; exogenous covariates (altitude, temperature; non-PPG, read before opening); ρ, ρ̃ descriptive only; L1 scope sentence | Cowork r2 S1, S2, L1; PI | v2.2 |
| 18 | rev 3 §4.3 | R vs R-thin on different test sets | paired comparison on the R-thin test set; direction predicted and worded in advance | PI B1–B3 | v2.2 |
| 19 | rev 5 §4.5 | K with m = 1 reported without a test | H-K1: second test of a fixed sequence (N_c − T_c/7, person-cluster bootstrap); X2 not used for m = 1; logistic bootstrap-failure rule | PI after 7cd7a1a | v2.2 |
| 20 | rev 2 §3 | B0-state: one fixed threshold per state | deleted; B0-state-matched kept | no fixed threshold was ever set before; setting one now would be a choice made from dev; B0-state-matched is already the oracle version that favours the baseline (PI 23-11) | rev 8 |

Data-handling events (no deviation of analysis, kept for the record): v2 change log (WildPPG ID mapping, DaLiA
confirmatory PPG removed from cache).

## 7. Items closed before freezing (PI)

1. Approved by PI 2026-11-23: K-intact infeasible; ρ̃ descriptive only (§4.2 b–c).
2. Approved by PI 2026-11-23: §3, B4 (ACI) kept as exploratory with its fixed parameters; B5 = R-sel; B2′ and B7
   dropped; B0-state deleted (rev 8).
3. Approved by PI 2026-11-23: the attribution table (§4.2, p_order + exogenous covariates), the H-R t test (§4.3)
   and the R vs R-thin wording.
4. Approved by PI 2026-11-23: H-K1 (§4.1b) and the bootstrap-failure rule (§4.2 d) (rev 6).
5. Dry run of the full confirmatory script on the 3 dev persons: **done** (rev 7, repeated in rev 8 at `d4ce6f2`,
   §4.7). 62 of 63 numbers match; the one mismatch is Monte Carlo and is explained. The two runs are byte-identical.
6. B0-state (§3): deleted by the PI 2026-11-23 (rev 8, deviation row 20).
7. Frozen 2026-11-23 (PI: "duyệt đóng băng"): commit of `prereg_v2_2.md`, tag `prereg-v2.2`. Next, in this order:
   external timestamp (OSF, URL recorded in `STATUS_repo.md`) → separate commit `CONFIRMATORY_PPG_OPEN = True` → one
   run (§4.7).
