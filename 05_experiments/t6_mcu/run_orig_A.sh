#!/bin/zsh
# T6 amendment 2: Measurement A with the ORIGINAL (dilated) model, after the dense A + B chain has finished.
# Flashes the pre-built original firmware (02_data/derived/t6/fw_orig, no rebuild) and replays e61, u7y, ngh once.
cd "$(dirname "$0")/../.."
LOG=02_data/derived/t6/AB_run.log; OUTLOG=02_data/derived/t6/AO_run.log; PORT=/dev/cu.usbserial-0001
until grep -q "^B end" "$LOG"; do sleep 30; done
{
  date '+AO flash %Y-%m-%d %H:%M:%S'
  . 06_firmware/idf_env.sh > /dev/null 2>&1
  (cd 02_data/derived/t6/fw_orig && python -m esptool --chip esp32 -p $PORT -b 460800 --before default_reset --after hard_reset write_flash @flash_args 2>&1 | grep -E "Hash of data|Wrote|rror")
  date '+AO start %H:%M:%S'
  05_experiments/t6_mcu/.venv/bin/python -u 05_experiments/t6_mcu/replay.py --port $PORT --mode A --prefix board_orig --no-rep2
  echo "AO exit $?"; date '+AO end %Y-%m-%d %H:%M:%S'
} > "$OUTLOG" 2>&1 < /dev/null
