# One-command reproduction of S1 so far:  make all
UV ?= uv run
.PHONY: all env test refs data pilot counts s1c spectra train_dev eval_dev sigma train_all wdev t4 s2 s2_pre s2_ana confirm_dry confirm paper
all: env test refs data pilot counts s1c wdev t4 s2
env:    ; uv sync --frozen
test:   ; $(UV) pytest -q 04_code/tests
refs:   ; $(UV) python 01_lit/verify_refs.py && $(UV) python 01_lit/make_bib.py
data:   ; ./02_data/scripts/download_dalia.sh && $(UV) python -I 02_data/scripts/stream_zip.py extract 02_data/raw/dalia/ppg+dalia.zip 02_data/raw/dalia/extracted PPG_FieldStudy/
pilot:  ; $(UV) python 05_experiments/pilot/run_pilot.py
counts: ; $(UV) python 05_experiments/pilot/count_ref_episodes.py
# S1c (development data only; WildPPG confirmatory PPG stays locked)
S1C = 05_experiments/s1c
SEEDS = 0 1 2 3 4
s1c: spectra train_dev eval_dev sigma train_all
spectra:   ; $(UV) python $(S1C)/build_spectra.py
train_dev: ; for s in $(SEEDS); do $(UV) python -W ignore $(S1C)/train_dev_folds.py $$s > $(S1C)/results/fcn_train_seed$$s.log 2>&1 & done; wait
eval_dev:  ; $(UV) python -W ignore $(S1C)/eval_dev.py fcn
sigma:     ; $(UV) python -W ignore $(S1C)/sigma_grid_ext.py
train_all: ; for s in $(SEEDS); do $(UV) python -W ignore $(S1C)/train_all_dalia.py $$s > $(S1C)/results/fcn_all_train_seed$$s.log 2>&1 & done; wait

# WildPPG dev (e61, u7y, ngh only; confirmatory PPG stays locked) -- needs WildPPG raw files, see 02_data
wdev:
	$(UV) python -W ignore $(S1C)/wildppg_dev_check.py zeroshot
	$(UV) python -W ignore $(S1C)/wildppg_dev_check.py finetune
	$(UV) python -W ignore $(S1C)/wildppg_inject_diag.py
	$(UV) python -W ignore $(S1C)/sqi_dev.py

T4 = 05_experiments/t4
t4:
	$(UV) python $(T4)/R1_vacuity.py
	$(UV) python $(T4)/R1_sim.py
	$(UV) python -W ignore $(T4)/dev_m_estimate.py

# S2 round 1 (Cowork review 22-11), DEV ONLY. s2_pre takes ~3 h on 4 threads (cache in 02_data/derived/s2_cache).
S2 = 05_experiments/s2
s2: s2_pre s2_ana
s2_pre:
	$(UV) python -W ignore $(S2)/precompute.py > $(S2)/results/precompute.log 2>&1
	$(UV) python -W ignore $(S2)/precompute_ramp.py > $(S2)/results/precompute_ramp.log 2>&1
s2_ana:
	$(UV) python -W ignore $(S2)/analysis.py > $(S2)/results/s2_analysis.log 2>&1
	$(UV) python $(S2)/m3_power.py > $(S2)/results/m3_power.log 2>&1
	$(UV) python $(S2)/m6_curve.py > $(S2)/results/m6_curve.log 2>&1
	$(UV) python -W ignore $(S2)/drift_extra.py > $(S2)/results/drift_extra.log 2>&1
	$(UV) python -W ignore $(S2)/controls.py > $(S2)/results/controls.log 2>&1
	$(UV) python -W ignore $(S2)/r3_checks.py > $(S2)/results/r3_checks.log 2>&1
	$(UV) python -W ignore $(S2)/s1e_attrib.py > $(S2)/results/s1e_attrib.log 2>&1
	$(UV) python -W ignore $(S2)/b_rchecks.py > $(S2)/results/b_rchecks.log 2>&1
	$(UV) python -W ignore $(S2)/b4_aci.py > $(S2)/results/b4_aci.log 2>&1
	$(UV) python -W ignore $(S2)/r5_porder_ma1.py > $(S2)/results/r5_porder_ma1.log 2>&1
	$(UV) python -W ignore $(S2)/r5_aci_bound.py > $(S2)/results/r5_aci_bound.log 2>&1
	$(UV) python -W ignore $(S2)/r5_km1_dev.py > $(S2)/results/r5_km1_dev.log 2>&1
	$(UV) python -W ignore $(S2)/r6_hk1.py > $(S2)/results/r6_hk1.log 2>&1

# Confirmatory pipeline (prereg v2.2 §4.7). Dry run on the dev IDs (twice) + regression check:
CONF = 05_experiments/confirm
confirm_dry:
	$(UV) python -W ignore $(CONF)/run_confirmatory.py --ids e61 u7y ngh --label dryrun_dev_r8_1
	$(UV) python -W ignore $(CONF)/run_confirmatory.py --ids e61 u7y ngh --label dryrun_dev_r8_2
	$(UV) python -W ignore $(CONF)/regression_check.py dryrun_dev_r8_1 dryrun_dev_r8_2
# Refused by the script while io_wildppg.CONFIRMATORY_PPG_OPEN = False (only after the prereg-v2.2 freeze commit):
confirm:
	$(UV) python -W ignore $(CONF)/run_confirmatory.py --ids an0 fex k2s kjd l38 n31 p5d p9p qm9 ssx trh tz8 w4p --label confirmatory

# T6 (feasibility, dev persons only; separate TF env 05_experiments/t6_mcu/.venv with tensorflow==2.21.0)
T6 := 05_experiments/t6_mcu
t6_pc:
	$(UV) python -W ignore $(T6)/export_inputs.py
	$(T6)/.venv/bin/python $(T6)/quantize.py
	$(UV) python -W ignore $(T6)/eval_int8_pc.py
t6_host:
	$(UV) python -W ignore $(T6)/gen_consts.py
	$(UV) python -W ignore $(T6)/host_check.py
	$(T6)/.venv/bin/python $(T6)/emulate_tflm.py
	$(UV) python -W ignore $(T6)/board_emul.py
	$(UV) python -W ignore $(T6)/compare_A.py emul
t6_fw:
	cd 06_firmware/h32_e2 && . ../idf_env.sh && idf.py build && idf.py size


# S3a manuscript: numbers, tables, figures, bibliography, PDF (LuaLaTeX via the TinyTeX inside H32), word counts
TEXBIN = $(firstword $(wildcard $(CURDIR)/.tinytex/bin/*))
paper:
	$(UV) python 07_manuscript/make_numbers.py
	$(UV) python 07_manuscript/make_tables.py
	$(UV) python 07_manuscript/figs/make_figs.py
	$(UV) python 07_manuscript/make_refs.py
	cd 07_manuscript && PATH="$(TEXBIN):$$PATH" latexmk main.tex
	$(UV) python 07_manuscript/count_words.py
