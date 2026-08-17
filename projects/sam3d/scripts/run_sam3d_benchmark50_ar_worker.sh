#!/usr/bin/env bash
set -euo pipefail

shard=${1:?usage: run_sam3d_benchmark50_ar_worker.sh SHARD GPU}
gpu=${2:?usage: run_sam3d_benchmark50_ar_worker.sh SHARD GPU}

workspace=${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}
project=${COSMOS_PROJECT:-${workspace}/cosmos-predict2.5-sam3d-musa}
runner=${COSMOS_RUNNER:-${workspace}/scripts/run_cosmos_sam3d_musa.sh}
eval_root=${WA_EVAL_ROOT:-/datassd/morka/cosmos-work/eval/sam3d_iter1000_bench50_ar93_640x480_121f_20260815}
checkpoint_dir=${WA_CHECKPOINT_DIR:-${workspace}/training-outputs/sam3d-v93-core15k-legacy4k-bs4-5k-norepa-20260814/cosmos_predict_v2p5_sam3d/video2world/sam3d-v93-core15k-legacy4k-bs4-5k-norepa-20260814/checkpoints/iter_000001000}
input=${eval_root}/inputs/shard_$(printf '%02d' "${shard}").jsonl
output=${eval_root}/videos
log=${eval_root}/logs/worker_$(printf '%02d' "${shard}").log
weight=${checkpoint_dir}/model_ema_bf16.pt

test -s "${input}"
test -s "${weight}"
mkdir -p "${output}" "${eval_root}/logs"

export COSMOS_OUTPUT_FPS=24
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MUSA_VISIBLE_DEVICES="${gpu}"

cd "${project}"
exec "${runner}" \
  examples/inference.py \
  -i "${input}" \
  -o "${output}" \
  --checkpoint-path "${weight}" \
  --experiment predict2_video2world_training_2b_sam3d_smoke \
  --context-parallel-size 1 \
  --disable-guardrails \
  --keep-going \
  --skip-existing-output \
  >>"${log}" 2>&1
