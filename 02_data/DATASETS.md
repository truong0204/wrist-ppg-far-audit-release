# H32 · Datasets (T2)

Checked: 2026-11-21. Every number below comes from the dataset's own README / data page / paper text
fetched on that date; the source is named in each row. `[CẦN KIỂM]` = not yet read from a primary source.
Raw data is never committed (`02_data/raw/` is git-ignored). Only download scripts and SHA256 live here.

## 1. Summary

| | PPG-DaLiA | WildPPG | GalaxyPPG |
|---|---|---|---|
| Role in H32 | Primary (pilot + main) | Proposed 2nd dataset (hour-scale claims) | Optional (consumer device) |
| Subjects | 15 (7 m / 8 f, age 30.60 ± 9.59) | 16 | 24 |
| Hours per subject | ≈ 2.5 h; **S6 only ≈ 1.5 h** (hardware issue) | ≈ 13.5 h (216 h total, one device location) | `[CẦN KIỂM: README in zip]` |
| Total hours (one wrist device) | ≈ 36 h (to be measured from files) | 216 h per location; 864 h over 4 locations | `[CẦN KIỂM]` |
| Wrist PPG | Empatica E4 BVP, 64 Hz | Custom device (MAX86141), green/IR/red PPG, 128 Hz, at head, sternum, wrist, ankle | Galaxy Watch 5 + Empatica E4 |
| Wrist accelerometer | E4, 32 Hz, 3 axes, units 1/64 g | 128 Hz, 3 axes (per device) | `[CẦN KIỂM]` |
| HR reference | Chest ECG (RespiBAN, 700 Hz); R-peaks auto-detected then **manually inspected and corrected**; label = mean instantaneous HR on 8 s windows, 2 s shift | Lead-I ECG at sternum, 128 Hz; R-peaks by Pan–Tompkins, IBIs outside 35–185 bpm removed (automatic, no manual correction stated) | Polar H10 chest ECG |
| Activity labels | Yes: 8 protocol activities (IDs 1–8) + transient (0), 4 Hz signal and start-time CSV | No strict protocol; activities described (walking, hiking, stairs, eating, transport). Per-sample activity labels `[CẦN KIỂM]` | "diverse activities in a semi-naturalistic setting"; label format `[CẦN KIỂM]` |
| License | CC BY 4.0 (UCI page; Q1 Plan check 17-11) `[re-check UCI page text]` | **CC BY-NC-SA 4.0** (paper, Appendix) — research use OK; non-commercial; derivatives share-alike | **CC BY 4.0** (Zenodo record metadata) |
| Access without request | Yes: `https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip` (HTTP 200) | Yes: public polybox share `https://polybox.ethz.ch/index.php/s/NWTuyNojU7aya1y` (WebDAV PROPFIND 207, no password), linked from `https://siplab.org/projects/WildPPG` | Yes: `https://zenodo.org/records/14635823/files/GalaxyPPG.zip` (HTTP 200, access_right=open) |
| Size | outer zip contains stored `data.zip` of 2,864,337,783 B | 16 × `WildPPG_Part_*.mat`, 1.11–1.38 GB each (≈ 19.6 GB) + `github_image.zip` 3.9 MB | `GalaxyPPG.zip` 481,594,233 B, md5 `d9a48bfcd07928fb22b96dc40cf2e7d4` (Zenodo) |
| Citation | Reiss et al. 2019, Sensors, 10.3390/s19143079 | Meier, Demirel, Holz 2024, NeurIPS D&B, 10.52202/079017-0073 | Park, Zheng, Lee 2025, Sci. Data, 10.1038/s41597-025-05152-z; data 10.5281/zenodo.14635823 |
| Usable for H32? | **Yes** | **Yes** (licence allows research; NC/SA obligations go into the paper's data statement) | **Yes** |

## 2. PPG-DaLiA details (from `PPG_FieldStudy_readme.pdf`, inside the archive)

- Protocol (README Table 2, nominal minutes): sitting 10, stairs 5, table soccer 5, cycling 8, driving 15,
  lunch 30, walking 10, working 20; transient periods (ID 0) between activities; ≈ 2.5 h total.
  S1, S2 went down the stairs only once.
- Synchronisation of E4 and RespiBAN: manual, by a double-tap gesture; clock drift corrected by deleting
  evenly spaced samples (max 1.2 s over > 9000 s, S9 and S10).
- `SX.pkl` keys: `activity` (4 Hz, IDs 0–8), `label` (HR-ECG per 8 s/2 s window), `questionnaire`,
  `rpeaks`, `signal` → `chest` (ACC, ECG, EDA, EMG, RESP, TEMP; EDA/EMG/Temp are dummy) and
  `wrist` (ACC, BVP, EDA, TEMP), `subject`.
- Windowing convention used by H32: window *i* covers [2*i*, 2*i*+8) s; BVP samples [128*i*, 128*i*+512),
  ACC samples [64*i*, 64*i*+256). Label *i* is the reference for window *i*.
- Archive order (streamed local headers): S10 first, then S11, … (S1 files are not first despite the
  folder entry). This is why the pilot uses S10–S12 (see `05_experiments/prereg.md`).
- `SX.pkl` is a Python-2 pickle. H32 loads it with a **restricted unpickler** that only admits numpy
  array reconstruction (`04_code/h32/io_dalia.py`); arbitrary classes are refused.

## 3. Download and checksums

| Dataset | Script | Checksum file |
|---|---|---|
| PPG-DaLiA | `02_data/scripts/download_dalia.sh` | `02_data/checksums/dalia_outer.sha256` (written when the download completes) |
| WildPPG | not downloaded in this round (≈ 19.6 GB; only needed after G0/G1) | — |
| GalaxyPPG | not downloaded in this round | Zenodo md5 above |

`02_data/scripts/stream_zip.py` reads members of the still-downloading archive by walking zip local
headers and checks each member's CRC-32 before writing it.

## 4. Open points

1. GalaxyPPG and WildPPG hours-per-subject and label format must be read from their own READMEs after
   download, not from the papers' abstracts.
2. WildPPG reference HR is automatic (no manual correction stated), unlike DaLiA. Episodes of
   HR-ECG > threshold may contain reference errors; H32 needs a reference-quality rule before using it.
3. DaLiA E4 "BVP" is the device's processed PPG output, not raw photodiode current.

## 5. WildPPG — update S1b (2026-11-21)

**Licence, verbatim from the two official sources (they differ):**
- Paper (arXiv 2412.17540, HTML full text, appendix): "The dataset uses a Creative Commons CC BY-NC-SA 4.0 license [60] while the code is released under GPL-3 [61]."
- Hugging Face release `eth-siplab/WildPPG` (dataset card YAML and API `cardData`): `license: mit`.
Both permit research use without a request. H32 follows the **stricter** terms (CC BY-NC-SA 4.0:
attribution, non-commercial, share-alike for derived data). The discrepancy is recorded here.

**Files.**
| Source | Content | Size | Checksum |
|---|---|---|---|
| polybox share (per ID) | 16 × `WildPPG_Part_<ID>.mat` (MATLAB 5.0), structs `sternum, head, ankle, wrist` with `acc_x/y/z, ecg (sternum only), ppg_g/ir/r, altitude, temperature`, each `{fs, descr, v}`; ACC/PPG/ECG at 128 Hz | 1.11–1.38 GB each | `02_data/checksums/wildppg.sha256` |
| Hugging Face `WildPPG.mat` | 16-cell arrays `data_bpm_values` (authors' cleaned HR, 8 s windows, 0 = removed), `data_ppg_*`, `data_imu_*`, `data_temp_*`, `data_altitude_values`; **no subject IDs** | 632,163,496 B | SHA256 `ed565ae798195889f9256f814b7233a87871e35113a9f5d84c18a93aaa757e6b` |

Reference HR (HF card): "obtained from ECG with Pan–Tompkins and cleaned further during motions".
Duration: e61 = 5,690,368 samples at 128 Hz = 12.35 h (measured); the card states "810 min (~13.5 h) each".
Index → ID mapping of the HF file is by recording length (`05_experiments/s1b/B_wildppg.py`).
Dev/confirmatory split by ID: prereg_v2 §1 (dev = e61, u7y, ngh).
