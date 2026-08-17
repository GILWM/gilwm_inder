#!/usr/bin/env bash
set -euo pipefail

root=/datassd/morka/cosmos-sam3d-work
checkpoint="$root/training-outputs/smoke-full-native-r1/cosmos_predict_v2p5_sam3d/video2world/smoke-full-native-r1/checkpoints/iter_000000002"
video="$root/inference-outputs/native-smoke-cp2/legacy4k_00000001_sam3d.mp4"

test -d "$root/cosmos-predict2.5-sam3d-musa"
test -d "$root/envs/cosmos-musa"
test -d "$root/weights/cosmos"
test -f "$root/weights/sam3/sam3.pt"
test -f "$root/weights/sam3d-objects/checkpoints/slat_generator.ckpt"
test -f "$root/offline/sam3d-offline-bundle-v3.tgz"
test -f "$checkpoint/model_ema_bf16.pt"
test -f "$video"
test -f "$root/cosmos-predict2.5-sam3d-musa/cosmos_predict2/_src/predict2/networks/sam3d_conditioned_dit.py"
test -f "$root/cosmos-predict2.5-sam3d-musa/cosmos_predict2/_src/predict2/datasets/local_datasets/sam3d_video_dataset.py"

if find \
  "$root/weights" \
  "$root/envs" \
  "$root/third_party" \
  "$root/cosmos-predict2.5-sam3d-musa" \
  -type l -printf '%p -> %l\n' | grep -q '/datassd/morka/cosmos-work'; then
  echo 'ERROR: found a runtime symlink back to the old workspace' >&2
  exit 1
fi

(
  cd "$root/weights/sam3d-objects/checkpoints"
  sha256sum -c SAM3D_OBJECTS_SHA256.txt
)

printf '%s  %s\n' \
  'ef4b99c54bd0bc2035ebb54dd229146b4bbb83126bffc1ed07e1b14082f566c4' \
  "$root/offline/sam3d-offline-bundle-v3.tgz" | sha256sum -c -

if find "$root" -type f -name '*.partial' -print -quit | grep -q .; then
  echo 'ERROR: found an unfinished .partial file' >&2
  exit 1
fi

echo INDEPENDENT_WORKSPACE_AUDIT_OK
du -sh \
  "$root/cosmos-predict2.5-sam3d-musa" \
  "$root/envs" \
  "$root/weights/cosmos" \
  "$root/weights/sam3" \
  "$root/weights/sam3d-objects" \
  "$root/third_party" \
  "$root/offline" \
  "$root/training-outputs" \
  "$root/inference-outputs"
