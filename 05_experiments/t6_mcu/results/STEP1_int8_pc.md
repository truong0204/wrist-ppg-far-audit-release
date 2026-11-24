# T6 step 1 · SpecFCN INT8 vs float on the PC (seed 0)

Generated from `results/int8_vs_float_pc.json` and `model/quant_meta.json` (scripts `export_inputs.py`, `quantize.py`, `eval_int8_pc.py`; `make t6_pc`). Same float64 spectra on both sides: only the quantisation of SpecFCN differs. INT8 run with the TFLite reference kernels (`BUILTIN_REF`), one window per invoke. No pass/fail criterion (PLAN_T6).

## Model

- TensorFlow 2.21.0; full-integer PTQ; calibration = 500 DaLiA windows from the 12 non-dev subjects.
- INT8 flatbuffer 18272 bytes (float32 flatbuffer 50044 bytes); ops: ADD, CONV_2D, RESHAPE.
- Input int8, scale 0.00392157, zero point -128; output int8 (256 logits), scale 0.0726071, zero point 66.
- Q0, Keras float vs PyTorch float, max |Δ logit|: 6.7e-06.

## HR and SQI per window (HMM σ = 3 bpm; grid step 0.8235 bpm)

| Set | Windows | \|ΔHR\| median / p99 / max (bpm) | Windows with ΔHR ≠ 0 | \|ΔSQI\| median / p99 / max | Monitored flag differs | MAE vs ECG float / INT8 |
|---|---|---|---|---|---|---|
| dalia_S10 | 5321 | 0.00 / 2.47 / 55.18 | 25.2 % | 0.0013 / 0.0063 / 0.0369 | 42 | 6.07 / 6.17 |
| dalia_S11 | 4521 | 0.00 / 3.29 / 84.82 | 26.7 % | 0.0013 / 0.0057 / 0.0244 | 49 | 14.80 / 15.00 |
| dalia_S12 | 3954 | 0.00 / 4.12 / 46.94 | 27.5 % | 0.0013 / 0.0066 / 0.0260 | 41 | 6.76 / 6.80 |
| wild_e61 | 22225 | 0.00 / 2.47 / 92.24 | 27.0 % | 0.0015 / 0.0067 / 0.0431 | 100 | — |
| wild_u7y | 23853 | 0.00 / 4.94 / 50.24 | 29.2 % | 0.0011 / 0.0064 / 0.0280 | 364 | — |
| wild_ngh | 26437 | 0.00 / 3.29 / 51.88 | 27.4 % | 0.0013 / 0.0063 / 0.0152 | 281 | — |

Float side vs the confirmatory dry-run cache (`dryrun_dev_r8_1`, seed 0): max |ΔHR| = 0, max |ΔSQI| ≤ 6e-8, states equal, for e61, u7y, ngh.

## Blocks and decisions (WildPPG dev; threshold "block score differs" |ΔS| > 0.41 bpm)

| Person | Blocks compared | Blocks differ | Δ in grid steps 0 / 1 / 2 / ≥ 3 | K: decisions differ / test blocks | K: τ per state float → INT8 (rest, light, vigorous) | R: decisions differ / test blocks | R: test blocks with different τ (max \|Δτ\|) |
|---|---|---|---|---|---|---|---|
| e61 | 69 | 22 | 47 / 22 / 0 / 0 | 0 / 51 | 103.3→104.1, 126.4→126.4, 143.6→143.6 | 0 / 48 | 10 (0.8) |
| u7y | 59 | 29 | 30 / 13 / 2 / 14 | 6 / 41 | 147.8→147.8, 74.5→74.5, 132.9→132.9 | 4 / 38 | 14 (15.6) |
| ngh | 75 | 40 | 35 / 18 / 1 / 21 | 1 / 57 | 114.8→114.8, 79.4→79.4, 92.6→85.2 | 4 / 54 | 26 (50.2) |
| **all 3** | 203 | 91 | | **7 / 149** | | **8 / 140** | |

No test block exists on one side only. These numbers are the quantisation effect alone (PC); the board adds float32 pre-processing (Measurement A).
