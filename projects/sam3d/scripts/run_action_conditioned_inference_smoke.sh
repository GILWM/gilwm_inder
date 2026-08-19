#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
project="${COSMOS_PROJECT:-${workspace}/cosmos-predict2.5-sam3d-musa}"
checkpoint_dir="${1:?usage: run_action_conditioned_inference_smoke.sh CHECKPOINT_DIR}"
input_json="${ACTION_INFERENCE_JSON:-${project}/projects/sam3d/configs/action_core15k_smoke.json}"
output_dir="${ACTION_INFERENCE_OUTPUT:-${workspace}/inference-outputs/action-core15k-smoke}"
model_pt="${checkpoint_dir}/model_ema_bf16.pt"
devices="${ACTION_INFERENCE_DEVICES:-0,1}"
gpus="${ACTION_INFERENCE_GPUS:-2}"

test -f "${checkpoint_dir}/model/.metadata"
test -s "${input_json}"
mkdir -p "${output_dir}"
cd "${project}"

if [[ ! -s "${model_pt}" ]]; then
  projects/sam3d/scripts/run_cosmos_sam3d_musa.sh \
    scripts/convert_distcp_to_pt.py "${checkpoint_dir}/model" "${checkpoint_dir}"
fi

export COSMOS_OUTPUT_FPS=24
export COSMOS_EXTRA_BIND_PATHS="${COSMOS_EXTRA_BIND_PATHS:-/datahdd}"
export MUSA_VISIBLE_DEVICES="${devices}"
projects/sam3d/scripts/run_cosmos_sam3d_musa.sh \
  -m torch.distributed.run --standalone --nproc-per-node="${gpus}" \
  examples/inference.py \
  -i "${input_json}" \
  -o "${output_dir}" \
  --checkpoint-path "${model_pt}" \
  --experiment predict2_video2world_inference_2b_sam3d_action \
  --disable-guardrails

video="${output_dir}/core15k_00000001_action_smoke.mp4"
projects/sam3d/scripts/run_cosmos_sam3d_musa.sh \
  projects/sam3d/scripts/verify_video_output.py \
  "${video}" --width 640 --height 480 --frames 93
echo "ACTION_CONDITIONED_INFERENCE_OK ${video} width=640 height=480 frames=93"
