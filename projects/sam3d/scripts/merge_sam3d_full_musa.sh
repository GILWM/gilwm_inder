#!/usr/bin/env bash
set -euo pipefail

workspace=/datassd/morka/cosmos-sam3d-work
cache=/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93
sidecars=/datahdd/mccxadmin/cosmos-sam3d-sidecars/core15k-legacy4k-v93
archive=/datahdd/mccxadmin/sam3d-full-output-v93.tar.zst
index=/datahdd/mccxadmin/cosmos-sam3d-native-index/core15k-legacy4k-v93.txt
expected=16874

test -s "$archive"
if [[ -n "${ARCHIVE_SHA256:-}" ]]; then
  printf '%s  %s\n' "$ARCHIVE_SHA256" "$archive" | sha256sum -c -
fi
mkdir -p "$sidecars" "$(dirname "$index")"
tar -C "$sidecars" --strip-components=1 -I zstd -xf "$archive"
find "$sidecars" -type f -name sam3d_objects.pt -printf '%P\n' \
  | sed 's#/sam3d_objects.pt$##' | sort >"$index"
test "$(wc -l <"$index")" = "$expected"

"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  "${workspace}/scripts/merge_sam3d_object_sidecars.py" \
  --base-cache-root "$cache" \
  --sidecar-root "$sidecars" \
  --output-root "$cache" \
  --max-objects 8 \
  --shape-tokens 256 \
  --allow-missing \
  --workers 16 \
  --overwrite

"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  "${workspace}/scripts/validate_sam3d_cache.py" "$cache" \
  --index "$index" \
  --workers 16 \
  --require-nonzero-tokens \
  --require-masks \
  --require-geometry \
  --require-sam3d-objects
echo "SAM3D_FULL_MUSA_MERGE_OK cache=$cache native_samples=$expected total_samples=18349 index=$index"
