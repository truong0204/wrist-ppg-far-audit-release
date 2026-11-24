# Data notice (H32)

Written 2026-11-23 (PI decision C(c), prompt S3b); licence wording updated 2026-11-24 (prompt S3f, A5).

## What this repository contains, and what it does not

- **No raw signals and no processed signals** of WildPPG or PPG-DaLiA are in this repository: no PPG, accelerometer,
  ECG, altitude or temperature series, and no per-window spectra. The download scripts and SHA256 checksums are in
  `02_data/`; the data must be obtained from their authors.
- The repository does contain **results derived from WildPPG**: `05_experiments/confirm/runs/*/results.json`,
  `conclusions.md`, the per-cell and per-person tables in `00_plan/EXPORT_*.md` and in the manuscript
  (`07_manuscript/`), the development-person summaries in `05_experiments/*/results/*.json`, and the figures made from
  them (for example `05_experiments/s2/figs/M2_blockmax_WildPPG.png`, `07_manuscript/figs/`).
- It also contains model weights trained on PPG-DaLiA (`05_experiments/t6_mcu/model/*.tflite`,
  `06_firmware/h32_e2/main/model_data.c`).

## Terms

- **Code: MIT licence** (`LICENSE`). The MIT licence applies to the code only, not to the data-derived files above.
- **Results derived from WildPPG are distributed under CC BY-NC-SA 4.0** (attribution, non-commercial use,
  share-alike for derived material). The licence statements differ between the publication and the dataset card; the
  stricter one is applied; not confirmed with the dataset authors. The two statements observed:

  | Source | Statement | URL | Read on |
  |---|---|---|---|
  | Dataset paper (arXiv:2412.17540, p. 16) | "The dataset uses a Creative Commons CC BY-NC-SA 4.0 license [60] while the code is released under GPL-3 [61]." | https://arxiv.org/abs/2412.17540 | 2026-11-21 (HTML); re-read in the PDF on 2026-11-23 |
  | Hugging Face release `eth-siplab/WildPPG` (dataset card) | `license: mit` | https://huggingface.co/datasets/eth-siplab/WildPPG | 2026-11-21 |

  CC BY-NC-SA 4.0 is the stricter of the two and is applied to every WildPPG-derived file listed above. The WildPPG
  authors were not asked (PI decision, prompt S3f, 2026-11-24).
- **Material derived from PPG-DaLiA** (model weights): PPG-DaLiA is listed under CC BY 4.0 on its UCI page
  (recorded on 2026-11-17 in `02_data/DATASETS.md`, marked there for a re-check of the page text); attribution: Reiss, Indlekofer, Schmidt & Van Laerhoven (2019), Sensors 19(14), 3079.

## Citation of the datasets

- WildPPG: Meier, Demirel & Holz (2024), "WildPPG: A Real-World PPG Dataset of Long Continuous Recordings",
  NeurIPS 2024 Datasets and Benchmarks.
- PPG-DaLiA: Reiss, Indlekofer, Schmidt & Van Laerhoven (2019), "Deep PPG: Large-Scale Heart Rate Estimation with
  Convolutional Neural Networks", Sensors 19(14), 3079.
