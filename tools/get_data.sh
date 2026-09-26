#!/usr/bin/env bash
# Fetch the standard 3DGS evaluation data into data/ (gitignored).
#   ./tools/get_data.sh scenes      tanks & temples + deep blending, COLMAP already run, 650 MB
#   ./tools/get_data.sh pretrained  Inria's trained .ply splats for every scene, 14.6 GB (renderer tests)
#   ./tools/get_data.sh mip360      Mip-NeRF 360, 12.5 GB (the other standard benchmark)
# Sizes verified 2026-09-22.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$root"
fetch() { # url dest
    [ -f "$2" ] && { echo "have $2"; return; }
    curl -L --fail --retry 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}
case "${1:-}" in
  scenes)
    fetch https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip "$root/tandt_db.zip"
    unzip -q -n "$root/tandt_db.zip" -d "$root" ;;
  pretrained)
    fetch https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip "$root/models.zip"
    unzip -q -n "$root/models.zip" -d "$root/pretrained" ;;
  mip360)
    fetch https://storage.googleapis.com/gresearch/refraw360/360_v2.zip "$root/360_v2.zip"
    unzip -q -n "$root/360_v2.zip" -d "$root/mip360" ;;
  *) sed -n '2,6p' "$0"; exit 1 ;;
esac
echo "done -> $root"
