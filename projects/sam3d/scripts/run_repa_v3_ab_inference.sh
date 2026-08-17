#!/usr/bin/env bash
set -euo pipefail

workspace=/datassd/morka/cosmos-sam3d-work
project=${workspace}/cosmos-predict2.5-sam3d-musa
checkpoint_dir=${1:?usage: run_repa_v3_ab_inference.sh CHECKPOINT_DIR OUTPUT_DIR [DEVICES]}
output_dir=${2:?usage: run_repa_v3_ab_inference.sh CHECKPOINT_DIR OUTPUT_DIR [DEVICES]}
devices=${3:-0,1}
input_jsonl=${workspace}/configs/repa_v3_ab3.jsonl
model_pt=${checkpoint_dir}/model_ema_bf16.pt

test -f "${checkpoint_dir}/model/.metadata"
test -s "${input_jsonl}"
if [[ ! -s "${model_pt}" ]]; then
  cd "${project}"
  "${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
    scripts/convert_distcp_to_pt.py "${checkpoint_dir}/model" "${checkpoint_dir}"
fi

export COSMOS_OUTPUT_FPS=24
export MUSA_VISIBLE_DEVICES="${devices}"
cd "${project}"
"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/inference.py \
  -i "${input_jsonl}" \
  -o "${output_dir}" \
  --checkpoint-path "${model_pt}" \
  --experiment predict2_video2world_training_2b_sam3d_smoke \
  --disable-guardrails

for sample in legacy4k_00000001 legacy4k_00000002 legacy4k_00000003; do
  video="${output_dir}/${sample}_sam3d.mp4"
  "${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
    "${workspace}/scripts/verify_video_output.py" \
    "${video}" --width 640 --height 480 --frames 121
done
