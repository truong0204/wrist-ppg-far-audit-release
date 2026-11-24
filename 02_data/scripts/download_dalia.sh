#!/usr/bin/env bash
# Download PPG-DaLiA (UCI id 495) into 02_data/raw/dalia/ (git-ignored) and record SHA256.
set -euo pipefail
# NOTE: the UCI server ignores HTTP Range (checked 2026-11-21), so a broken download restarts from 0.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/raw/dalia"
URL="https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
mkdir -p "$DEST"
if [ ! -f "$DEST/ppg+dalia.zip" ]; then
  uv run --project "$HERE/.." python -I "$(dirname "$0")/fetch_resume.py" "$URL" "$DEST/ppg+dalia.zip.part" 30
  mv "$DEST/ppg+dalia.zip.part" "$DEST/ppg+dalia.zip"
fi
( cd "$DEST" && shasum -a 256 ppg+dalia.zip ) | tee "$HERE/checksums/dalia_outer.sha256.actual"
if [ -f "$HERE/checksums/dalia_outer.sha256" ]; then
  diff <(awk '{print $1}' "$HERE/checksums/dalia_outer.sha256") <(awk '{print $1}' "$HERE/checksums/dalia_outer.sha256.actual") \
    && echo "SHA256 OK" || { echo "SHA256 MISMATCH"; exit 1; }
fi
