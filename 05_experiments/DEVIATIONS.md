# H32 · Deviations from prereg v2.2 (confirmatory run)

No deviation in analysis code or numbers. Three departures in procedure and wording are listed below.

- Prereg: `05_experiments/prereg_v2_2.md`, SHA256 `35474f6a65fbc563543231e8383cf09722d96908839088e5205862e2a5dfd7b4`,
  freeze commit `2decaae` (2026-11-23 09:05:37 +07), tag `prereg-v2.2`.
- Opening commit `d402033` (2026-11-23 13:23:33 +07). Run `make confirm`, once, 13:23:43 → 14:16:25 +07, at HEAD
  `d402033`; `results.json` SHA256 `2460459493a626f8e6205a1fbdc1080278378e990ec8c5d5ac013cfee77cffb2`.
- The script hashes at the run equal §4.7 (`confirm.py` `58624dde…`, `run_confirmatory.py` `1a49ee40…`,
  `method.py` `17e84a0a…`). The automatic deviation list of the script is empty. n = 13 in every test; no
  unreadable file; no rerun.
- Approved by the PI on 2026-11-23: D1, D2 and D3 (`00_plan/REVIEW_S2_confirmatory_2026-11-23.md` §3); for D3 the
  PI chose filling F2 on 2026-11-23.

---

## D1 · External timestamp: OpenTimestamps instead of OSF

| Field | Content |
|---|---|
| Code | D1 |
| Date/time | 2026-11-23 13:23 +07 (OpenTimestamps submission) |
| Commits | `3df94f2` (proof file `05_experiments/prereg_v2_2.md.ots`), `d402033` (opening) |
| Prereg says | §7 item 7: "external timestamp (OSF, URL recorded in `STATUS_repo.md`) → separate commit `CONFIRMATORY_PPG_OPEN = True` → one run (§4.7)". |
| Done instead | OpenTimestamps, by the PI's decision. Only the SHA256 of `prereg_v2_2.md` was sent to four calendars (a.pool / b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com). |
| Effect on conclusions | None on the numbers. It affects only what an outside reader can verify about timing (below). |
| PI approval | 2026-11-23 |

**Times (+07):**

| Event | Time | Source |
|---|---|---|
| Freeze commit `2decaae` | 09:05:37 | git |
| GitHub creates branch `main` at `e51f3a8` (`2decaae` is its ancestor) | 10:11:43 | GitHub activity API (`timestamps/github_activity.json`, `branch_creation`, 03:11:43Z) |
| OpenTimestamps submission | 13:23:11 | `ots stamp` output |
| GitHub receives push of `3df94f2` (.ots file) | 13:23:27 | GitHub activity API |
| Opening commit `d402033` | 13:23:33 | git |
| GitHub receives push of `d402033` | 13:23:36 | GitHub activity API |
| Run starts | 13:23:43 | `meta.json` |
| Run ends | 14:16:25 | `make` wrapper |
| Bitcoin block 977165 containing both calendar transactions | 14:23:51 | mempool.space API, `block_time` 1795418631 |

**`ots upgrade` and `ots verify` (2026-11-23 14:45 +07):** "Timestamp not complete". The calendars alice and bob
report transactions `6e9172ae4f91968da836023625848f7ec2ced3ff39402de688acffe9554f937e` and
`0bc9d9b0ed21207e4fcceca784cabc55c1a60a4a3101525a226317bdaac23bf5`, "waiting for 6 confirmations"; finney reports
"pending confirmation". Both transactions are in block 977165 (block hash
`000000000000000000014641e7b8d308b6a5eaec5772d150ceed4a4f3720e600`), looked up on mempool.space; chain tip at the
query was 977169.

**Second `ots upgrade` (2026-11-23 15:01 +07, chain tip 977171):** "Success! Timestamp complete". The proof now holds
`BitcoinBlockHeaderAttestation(977165)` (from the bob calendar); alice and finney are still listed as pending.
- `ots verify` did **not** run to completion: "Could not connect to Bitcoin node" (no local Bitcoin node on this
  machine). The block height and time are therefore not reported by `ots verify` itself.
- Manual check instead: the merkle root that the proof commits to at height 977165 is
  `e8b0d1e5af59e0fc605218c2a25e36ac061a4558b7d2d750ab1bb757a3fbd070` (read from the `.ots` file with the
  `opentimestamps` library); the header of block 977165 on mempool.space has the same merkle root, block hash
  `000000000000000000014641e7b8d308b6a5eaec5772d150ceed4a4f3720e600`, block time 1795418631 = **2026-11-23
  14:23:51 +07**.
- A full `ots verify` needs a Bitcoin node (or a later client run against one); the upgraded proof is committed.

**What the timestamps prove.**
- The OpenTimestamps proof does **not** show that the plan existed before opening. Its Bitcoin block (14:23:51)
  is later than the opening commit (13:23:33) and later than the end of the run (14:16:25).
- The third-party record that predates opening is GitHub's: branch `main` was created on GitHub at 10:11:43 +07
  with head `e51f3a8`, whose history contains the freeze commit `2decaae` (committed 09:05:37).
- No separate GitHub record of the tag push `prereg-v2.2` was returned: `activity?ref=refs/tags/prereg-v2.2` gave
  an empty list (`timestamps/github_activity_tag_prereg-v2.2.json`). The events API returned 2 events
  (`timestamps/github_events.json`).
- Raw JSON, as returned: `05_experiments/timestamps/` (query time in `queried_at_utc.txt`). GitHub keeps these
  records for about 90 days.
- Neither OSF nor OpenTimestamps nor GitHub can show that confirmatory data were not looked at before a time; they
  only date the plan.

---

## D2 · Attribution row 3: the add-on sentence about dTime

| Field | Content |
|---|---|
| Code | D2 |
| Date/time | 2026-11-23 14:16 +07 (run); decision 2026-11-23 |
| Commits | script `d4ce6f2` (`confirm.py` lines 742–744), run at `d402033` |
| Prereg says | §4.2: in row 3, "if the dAlt CI of sensitivity model 2 (with dTime) contains 0, this is added: 'altitude change and time elapsed since calibration could not be separated.'" §4.2 d: a model with more than 5 % of bootstrap draws dropped is reported only by the exceedance rate per dAlt tercile, with no coefficient or CI. |
| What happened | Sensitivity model 2 (dAlt + dTime) dropped 288 of 2,000 draws (14.4 %); dTime alone dropped 213 (10.65 %). By §4.2 d neither has a reportable CI. The prereg does not say how the add-on sentence is decided in that case. The script still compared the sensitivity-2 dAlt CI with 0; it did not contain 0, so no add-on was printed in `conclusions.md`. |
| Decision | That CI is not used in either direction. The paper adds, verbatim: |
| Effect on conclusions | The row-3 sentence in `conclusions.md` is unchanged (primary model only). The added sentence states that altitude change and elapsed time were not separated. |
| PI approval | 2026-11-23 |

Sentence added to the paper:

> "The model intended to separate altitude change from elapsed time did not meet the pre-specified stability rule
> (14.4 % of bootstrap draws dropped; limit 5 %). Altitude change and elapsed time since calibration were therefore
> not separated in these data. This case was not covered by the analysis plan; the sentence was added after
> unblinding."

- The log does not record the reason for each dropped draw (no variation of the outcome, optimiser failure, or
  |coefficient| > 20); `results.json` stores only the counts (`covariates.bootstrap_dropped`). Why the two models
  with dTime dropped draws is therefore not known. The run was not repeated to find out.

---

## D3 · H-R sentence: missing clause on sensitivity cost and masking

| Field | Content |
|---|---|
| Code | D3 |
| Date/time | Present since the script of rev 7 (`9b99fd5`/`caf5bff`, dry runs `dryrun_dev_1/2`) and rev 8 (`d4ce6f2`, dry runs `dryrun_dev_r8_1/2`); found after opening, 2026-11-23 |
| Commits | `confirm.py` at `d4ce6f2`, lines 756–759 (first written in `9b99fd5`, already without the clause) |
| Prereg says | §4.3, "H-R not rejected": "Rolling calibration kept the long-run rate consistent with its expectation bound of 0.857/h (mean of persons x, 90 % CI …): not refuted at n = 13, which is not the same as "the bound holds". This came at a sensitivity cost of … and with masking of … after an event. Rolling calibration gives only an expectation bound, not the per-cell tolerance guarantee." |
| What the script printed | The same sentence without "This came at a sensitivity cost of … and with masking of … after an event." The regression check of the dry runs compared numbers, not sentences, so this was not caught before opening. |
| Decision | The paper uses the full prereg sentence, filled with numbers already computed in the single run. No script change, no rerun. |
| Effect on conclusions | None on the branch or the numbers; the clause adds numbers already in the export (§8, §9). |
| PI approval | Handling approved 2026-11-23. **PI chose F2 on 2026-11-23.** |

**Sentence used in the paper (F2, chosen by the PI on 2026-11-23):**

> "Rolling calibration kept the long-run rate consistent with its expectation bound of 0.857/h (mean of persons
> 0.88, 90 % CI 0.74–1.03): not refuted at n = 13, which is not the same as "the bound holds". This came at a
> sensitivity cost of 0.243 / 0.373 / 0.453 vs 0.371 / 0.475 / 0.502 for K (+20 / +30 / +40 bpm) and with masking
> of a second event 2–7 state-blocks after a first one (detection 0.192–0.261 vs 0.272–0.391; none at h = 1 and
> h ≥ 8). Rolling calibration gives only an expectation bound, not the per-cell tolerance guarantee."

The inserted clause is the PI's text verbatim; the rest is the sentence of `conclusions.md`, unchanged.

- (a) The masking at h = 2–7 is a mechanical consequence of R's window: τ_{k,j} = max(S_{k,j−7}, …, S_{k,j−2})
  (prereg §2), so a first event in block j0 enters the threshold of exactly the test blocks j0 + 2 … j0 + 7. The
  simulation only measures its size, at Δ = +30.
- (b) The `_self` variant (threshold estimated on the injected stream itself) is reported in the tables only, not
  in the sentence.

The prereg sentence did not say which Δ, which comparator, which state or which h to use. The candidate fillings
that were considered are kept below, with their sources in `results.json`, so that the rejected options are
visible:

Sensitivity (key `sensitivity.summary["<Δ>|all"]`; all states; threshold from the clean stream; events per Δ:
1849 / 1852 / 1871):

| Δ | `sens_R` | `sens_K` | `sens_B0` | `sens_R_self` |
|---|---|---|---|---|
| +20 | 0.243 | 0.371 | 0.101 | 0.123 |
| +30 | 0.373 | 0.475 | 0.164 | 0.147 |
| +40 | 0.453 | 0.502 | 0.266 | 0.161 |

Masking (key `masking.R["<h>"]`, fields `after` / `no_prior`; the masking simulation uses Δ = +30, `confirm.masking`
default `delta=30`; n = 7240 per h):

| h | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8–14 |
|---|---|---|---|---|---|---|---|---|
| after | 0.444 | 0.261 | 0.237 | 0.221 | 0.208 | 0.197 | 0.192 | equal to no_prior |
| no_prior | 0.444 | 0.391 | 0.347 | 0.321 | 0.297 | 0.281 | 0.272 | (0.277–0.413) |

Fillings considered (all numbers above, nothing new); **F2 chosen**, F1, F3 and F4 not used:

- **F1 · one Δ, one h.** Δ = +30 (the Δ of the masking simulation), comparator K, h = 2:
  "This came at a sensitivity cost of 0.373 vs 0.475 for K (+30 bpm events) and with masking of a second event
  2 state-blocks after a first one (detection 0.261 vs 0.391)."
- **F2 · all Δ, h range.** Comparator K at Δ = +20/+30/+40, masking over h = 2–7:
  "This came at a sensitivity cost of 0.243 / 0.373 / 0.453 vs 0.371 / 0.475 / 0.502 for K (+20 / +30 / +40 bpm)
  and with masking of a second event 2–7 state-blocks after a first one (detection 0.192–0.261 vs 0.272–0.391;
  none at h = 1 and h ≥ 8)."
- **F3 · as F1 or F2 but comparator B0** (fixed population threshold) instead of K. Note: R is above B0 at every Δ,
  so "cost" would not describe this comparison.
- **F4 · point to the tables** without numbers in the sentence: "This came at the sensitivity cost and masking
  after an event shown in Table X."

PI's answers (2026-11-23): comparator K; all three Δ; h = 2–7 as a range; `_self` in the tables only.

---

## Correction C1 (23-11-2026) · n of the masking simulation per rule

Approved by the PI on 2026-11-23 (prompt S3b, item B1). A correction of this file, not a deviation from the plan:
no code, result or conclusion changes. The text above is left as written; this section supersedes one number in it.

- **What was wrong.** D3 above states "n = 7240 per h" for the masking table of rule R. In `results.json` the number
  of simulated cases differs by rule and is the same for every h within a rule:

  | Rule | n per h | Key in `results.json` |
  |---|---|---|
  | K | 7240 | `masking.K["<h>"].n` |
  | ACI | 7240 | `masking.ACI["<h>"].n` |
  | R | **6490** | `masking.R["<h>"].n` |
  | R-sel | 6490 | `masking.Rsel["<h>"].n` |
  | R-thin | 2630 | `masking.Rthin["<h>"].n` |

  The R masking values in D3 (`after`, `no_prior`) are correct; only their n was wrong (6490, not 7240).
- **Cause.** `00_plan/EXPORT_S2_confirmatory_for_Cowork.md` §9 printed one n for all rows, taken from rule K
  (`05_experiments/confirm/make_export.py`, line 261: `m['K'][hs[0]]['n']`), and D3 copied it. The Cowork review
  `00_plan/REVIEW_S2_confirmatory_2026-11-23.md` does not mention the n of the masking table, so it did not catch this.
- **Reading.** R (and R-sel, R-thin) are undefined on the first blocks of each cell, so K and R are evaluated on
  different block sets. No common-support recomputation was performed (PI decision).
- **In the paper.** The masking table gives n per rule with that sentence.
- The export file and `make_export.py` are left unchanged (they are the record of what was sent); this correction is
  the reference.
