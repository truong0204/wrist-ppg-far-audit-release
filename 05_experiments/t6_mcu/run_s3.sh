#!/bin/zsh
# T6 on the ESP32-S3 (PLAN_T6 S3 plan items 4-5, amendment 4): A-S3-1 (first 1,800 windows of each dev person) and
# Measurement B, both models. Firmware is flashed from the frozen binaries (02_data/derived/t6/fw_s3_{orig,dense}); before
# each stage the board is pinged and the model size is checked (orig 18,272 B, dense 24,472 B).
# Log: 02_data/derived/t6/S3_run.log
cd "$(dirname "$0")/../.."
P=${PORT:?set PORT to the board UART port /dev/cu.usbmodem...}   # port name not recorded: it contains the USB serial number
PY=05_experiments/t6_mcu/.venv/bin/python; R=05_experiments/t6_mcu/replay.py
flash() { (cd 02_data/derived/t6/fw_s3_$1 && . ../../../../06_firmware/idf_env.sh >/dev/null 2>&1 && \
          python -m esptool --chip esp32s3 -p $P -b 460800 --before default_reset --after hard_reset write_flash @flash_args | grep -E "Wrote|verified"); }
check() { $PY $R --port $P --mode ping > 02_data/derived/t6/ping_s3.json 2>&1; grep -q "\"model_bytes\": $1" 02_data/derived/t6/ping_s3.json || { echo "WRONG FIRMWARE (want model_bytes $1)"; cat 02_data/derived/t6/ping_s3.json; exit 3; }; echo "ping ok, model_bytes $1"; }
{
  set -e
  date '+start %F %T'
  echo "== original model (expected on the board after bring-up)"; check 18272
  date '+A-S3-1 orig %T'; $PY -u $R --port $P --mode A --prefix s3A_orig --a-windows 1800 --no-rep2
  date '+B orig %T';      $PY -u $R --port $P --mode B --runs 10 --b-windows 100 --b-ids e61 --b-out B_s3_orig
  echo "== dense model"; flash dense; check 24472
  date '+A-S3-1 dense %T'; $PY -u $R --port $P --mode A --prefix s3A_dense --a-windows 1800 --no-rep2
  date '+B dense %T';      $PY -u $R --port $P --mode B --runs 10 --b-windows 1800 --b-out B_s3_dense
  date '+end %F %T'
} > 02_data/derived/t6/S3_run.log 2>&1 < /dev/null
