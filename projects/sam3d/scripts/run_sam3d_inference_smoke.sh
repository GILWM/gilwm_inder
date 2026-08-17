#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
project="${workspace}/cosmos-predict2.5-sam3d-musa"
checkpoint_dir="${1:?usage: run_sam3d_inference_smoke.sh CHECKPOINT_DIR}"
input_json="${SAM3D_INFERENCE_JSON:-${workspace}/configs/sam3d_inference_smoke.json}"
output_dir="${SAM3D_INFERENCE_OUTPUT:-${workspace}/inference-outputs/native-smoke}"
model_pt="${checkpoint_dir}/model_ema_bf16.pt"
inference_gpus="${SAM3D_INFERENCE_GPUS:-2}"
inference_devices="${SAM3D_INFERENCE_DEVICES:-0,1}"

for required in "${checkpoint_dir}/model/.metadata" "${input_json}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required inference artifact: ${required}" >&2
    exit 2
  fi
done

cd "${project}"
if [[ ! -s "${model_pt}" ]]; then
  "${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
    scripts/convert_distcp_to_pt.py "${checkpoint_dir}/model" "${checkpoint_dir}"
fi

export COSMOS_OUTPUT_FPS=24
# A 121-frame video has 31 temporal latent frames.  At 480x640 resolution the
# official find_split() can only consume one additional factor of two from the
# height dimension; CP=4/8 would therefore split width, which Cosmos rejects to
# protect output quality.  CP=2 is the largest valid context-parallel group for
# this exact smoke geometry and comfortably fits the 2B model on 80 GiB cards.
export MUSA_VISIBLE_DEVICES="${inference_devices}"
"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  -m torch.distributed.run --standalone --nproc-per-node="${inference_gpus}" \
  examples/inference.py \
  -i "${input_json}" \
  -o "${output_dir}" \
  --checkpoint-path "${model_pt}" \
  --experiment predict2_video2world_training_2b_sam3d_smoke \
  --disable-guardrails

video="${output_dir}/legacy4k_00000001_sam3d.mp4"
if [[ ! -s "${video}" ]]; then
  echo "Inference did not produce a non-empty video: ${video}" >&2
  exit 3
fi
"${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  "${workspace}/scripts/verify_video_output.py" \
  "${video}" --width 640 --height 480 --frames 121
echo "SAM3D_VIDEO_INFERENCE_OK ${video} width=640 height=480 frames=121"
