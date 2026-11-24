#!/usr/bin/env bash
# Download WildPPG (Meier, Demirel, Holz, NeurIPS 2024 D&B; CC BY-NC-SA 4.0) from the public ETH polybox
# share linked on https://siplab.org/projects/WildPPG. Raw data stays in 02_data/raw/wildppg/ (git-ignored).
# Dev/confirmatory split is fixed in 05_experiments/prereg_v2.md §1 (by ID, before any download).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; DEST="$HERE/raw/wildppg"; mkdir -p "$DEST"
BASE="https://polybox.ethz.ch/public.php/dav/files/NWTuyNojU7aya1y/data"
IDS="e61 u7y ngh an0 fex k2s kjd l38 n31 p5d p9p qm9 ssx trh tz8 w4p"   # dev first
for id in $IDS; do
  f="WildPPG_Part_${id}.mat"
  [ -f "$DEST/$f" ] && continue
  curl -L --fail --retry 5 -C - -A "research-client/0.1" -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"; echo "done $f"
done
( cd "$DEST" && shasum -a 256 WildPPG_Part_*.mat ) > "$HERE/checksums/wildppg.sha256"
echo "all done"
