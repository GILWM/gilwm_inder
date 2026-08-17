#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
project="${workspace}/cosmos-predict2.5-sam3d-musa"
checkpoint_dir="${1:?usage: run_sam3d_ar93_smoke.sh CHECKPOINT_DIR}"
input_json="${workspace}/configs/sam3d_ar93_smoke.json"
output_dir="${workspace}/inference-outputs/ar93-smoke"
model_pt="${checkpoint_dir}/model_ema_bf16.pt"

test -s "${model_pt}"
test -s "${input_json}"
mkdir -p "${output_dir}"
cd "${project}"

export COSMOS_OUTPUT_FPS=24
export MUSA_VISIBLE_DEVICES="${SAM3D_AR_DEVICES:-0,1}"
"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/inference.py \
  -i "${input_json}" \
  -o "${output_dir}" \
  --checkpoint-path "${model_pt}" \
  --experiment predict2_video2world_training_2b_sam3d_smoke \
  --disable-guardrails

video="${output_dir}/legacy4k_00000001_sam3d_ar93x2.mp4"
"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  "${workspace}/scripts/verify_video_output.py" \
  "${video}" --width 640 --height 480 --frames 185
echo "SAM3D_AR93_SMOKE_OK ${video} chunks=2 chunk_size=93 overlap=1 frames=185"
