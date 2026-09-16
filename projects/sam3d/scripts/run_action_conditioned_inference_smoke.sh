#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:?set COSMOS_SAM3D_WORKSPACE}"
project="${COSMOS_PROJECT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
checkpoint_dir="${1:?usage: run_action_conditioned_inference_smoke.sh CHECKPOINT_DIR}"
input_json="${ACTION_INFERENCE_JSON:?set ACTION_INFERENCE_JSON to a prepared input JSON}"
output_dir="${ACTION_INFERENCE_OUTPUT:-${workspace}/inference-outputs/action-core15k-smoke}"
model_pt="${checkpoint_dir}/model_ema_bf16.pt"
devices="${ACTION_INFERENCE_DEVICES:-0,1}"
gpus="${ACTION_INFERENCE_GPUS:-2}"
experiment="${SAM3D_ACTION_INFERENCE_EXPERIMENT:-predict2_video2world_inference_2b_sam3d_action}"

test -s "${input_json}"
mkdir -p "${output_dir}"
cd "${project}"

if [[ ! -s "${model_pt}" ]]; then
  test -f "${checkpoint_dir}/model/.metadata"
  projects/sam3d/scripts/run_cosmos_sam3d_musa.sh \
    scripts/convert_distcp_to_pt.py "${checkpoint_dir}/model" "${checkpoint_dir}"
fi

export COSMOS_EXTRA_BIND_PATHS="${COSMOS_EXTRA_BIND_PATHS:-}"
export SAM3D_BASE_CHECKPOINT="${checkpoint_dir}"
export MUSA_VISIBLE_DEVICES="${devices}"
projects/sam3d/scripts/run_cosmos_sam3d_musa.sh \
  -m torch.distributed.run --standalone --nproc-per-node="${gpus}" \
  examples/inference.py \
  -i "${input_json}" \
  -o "${output_dir}" \
  --checkpoint-path "${model_pt}" \
  --experiment "${experiment}" \
  --disable-guardrails

video="${ACTION_INFERENCE_VIDEO:?set ACTION_INFERENCE_VIDEO to the expected output MP4}"
projects/sam3d/scripts/run_cosmos_sam3d_musa.sh \
  projects/sam3d/scripts/verify_video_output.py \
  "${video}" --width 640 --height 480 --frames 93
echo "ACTION_CONDITIONED_INFERENCE_OK ${video} width=640 height=480 frames=93"
