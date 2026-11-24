#!/bin/zsh
# T6 amendment 3: when the dense A on the ESP32-D0WD-V3 has finished, run the reference B (n = 3 runs after reset,
# first 100 windows of e61, dense model) and the A comparison. Log: 02_data/derived/t6/after_A.log
cd "$(dirname "$0")/../.."
while pgrep -f "replay.py --port /dev/cu.usbserial-0001 --mode A" > /dev/null; do sleep 30; done
{
  date '+after A %Y-%m-%d %H:%M:%S'; ls -l 02_data/derived/t6/board_*.npz
  if [ -f 02_data/derived/t6/board_e61_rep2.npz ]; then
    05_experiments/t6_mcu/.venv/bin/python -u 05_experiments/t6_mcu/replay.py --port /dev/cu.usbserial-0001 --mode B --runs 3 --b-windows 100 --b-ids e61 --b-out B_ref_esp32
    echo "Bref exit $?"; date '+Bref end %H:%M:%S'
  else
    echo "A incomplete: board_e61_rep2.npz missing; B not run"
  fi
} > 02_data/derived/t6/after_A.log 2>&1 < /dev/null
