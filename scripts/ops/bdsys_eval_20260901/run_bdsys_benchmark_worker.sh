#!/usr/bin/env bash
set -euo pipefail

shard=${1:?usage: run_bdsys_benchmark_worker.sh SHARD GPU}
gpu=${2:?usage: run_bdsys_benchmark_worker.sh SHARD GPU}

container_workspace=/datassd/morka/cosmos-sam3d-work
container_project=${COSMOS_PROJECT:-${container_workspace}/cosmos-predict2.5-sam3d-noaction-20260827}
host_workspace=/bdsys_hdd/datassd/morka/cosmos-sam3d-work
runner=${COSMOS_RUNNER:-${host_workspace}/scripts/run_cosmos_sam3d_musa_bdsys.sh}
eval_root=${WA_EVAL_ROOT:?WA_EVAL_ROOT must be the container-visible /datassd path}
checkpoint_dir=${WA_CHECKPOINT_DIR:?WA_CHECKPOINT_DIR must be the container-visible /datassd path}
host_eval_root="/bdsys_hdd${eval_root}"
host_checkpoint_dir="/bdsys_hdd${checkpoint_dir}"
input=${eval_root}/inputs/shard_$(printf '%02d' "${shard}").jsonl
host_input=${host_eval_root}/inputs/shard_$(printf '%02d' "${shard}").jsonl
output=${eval_root}/videos
log=${host_eval_root}/logs/worker_$(printf '%02d' "${shard}").log
weight=${checkpoint_dir}/model_ema_bf16.pt

test -s "${host_input}"
test -s "${host_checkpoint_dir}/model_ema_bf16.pt"
mkdir -p "${host_eval_root}/videos" "${host_eval_root}/logs"

export COSMOS_OUTPUT_FPS=24
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MUSA_VISIBLE_DEVICES="${gpu}"
export SAM3D_CONDITION_PROOF=1

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
