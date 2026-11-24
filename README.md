# wrist-ppg-far-audit-release

Analysis plan, deviations, timestamp records, code and results of the study "Per-wearer false-alarm guarantees for
wrist-PPG heart-rate alarms did not hold under outdoor free-living wear: a pre-specified audit on WildPPG" (Nguyen Xuan
Truong, FPT School of Business & Technology, FPT University). The files are copied unchanged from the project
repository, whose version-control history (tag `prereg-v2.2`, freeze commit `2decaae`, opening commit `d402033`) is
not public and is available to the editor and reviewers on request. An archival copy of the project files at
submission, including the scripts that generate the numbers, tables and figures of the paper, is at Zenodo:
DOI 10.5281/zenodo.24485648.

| Path | Content |
|---|---|
| `05_experiments/prereg_v2_2.md` | the analysis plan, fixed before unblinding (SHA256 `35474f6a65fbc563543231e8383cf09722d96908839088e5205862e2a5dfd7b4`) |
| `05_experiments/DEVIATIONS.md` | departures from the plan and corrections |
| `05_experiments/timestamps/` | records of when the plan existed (GitHub activity and events) |
| `04_code/` | analysis package `h32` and its tests (`make test`) |
| `05_experiments/` | experiment scripts and their results: pilot, development analyses (`s1b`, `s1c`, `s2`, `t4`), the confirmatory run (`confirm/runs/confirmatory/results.json`, SHA256 `2460459493a626f8e6205a1fbdc1080278378e990ec8c5d5ac013cfee77cffb2`), and the microcontroller study (`t6_mcu`, with the INT8 models) |
| `06_firmware/` | ESP-IDF firmware of the microcontroller study (toolchain not included) |
| `02_data/` | download scripts, dataset notes and SHA256 checksums only; **no data** |

Environment: `uv sync --frozen` (`pyproject.toml`, `uv.lock`); targets in the `Makefile`. The raw data must be
obtained from the dataset authors (`02_data/scripts/`). Some targets and scripts refer to files that are only in the
archival copy (manuscript, notes).

Licences: code MIT (`LICENSE`); results derived from WildPPG CC BY-NC-SA 4.0, material derived from PPG-DaLiA under
the terms of PPG-DaLiA (`05_experiments/DATA_NOTICE.md`). No raw or processed signals are included.
