#!/usr/bin/env bash
set -euo pipefail

eval_root=${1:?usage: run_action_benchmark50_worker.sh EVAL_ROOT CHECKPOINT_DIR SHARD GPU}
checkpoint_dir=${2:?usage: run_action_benchmark50_worker.sh EVAL_ROOT CHECKPOINT_DIR SHARD GPU}
shard=${3:?usage: run_action_benchmark50_worker.sh EVAL_ROOT CHECKPOINT_DIR SHARD GPU}
gpu=${4:?usage: run_action_benchmark50_worker.sh EVAL_ROOT CHECKPOINT_DIR SHARD GPU}

workspace=${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}
project=${COSMOS_PROJECT:-${workspace}/action-alignment-audit-code-20260820}
runner=${project}/projects/sam3d/scripts/run_cosmos_sam3d_musa.sh
input=${eval_root}/inputs/shard_$(printf '%02d' "${shard}").jsonl
output=${eval_root}/videos
weight=${checkpoint_dir}/model_ema_bf16.pt

test -s "${input}"
test -s "${weight}"
mkdir -p "${output}" "${eval_root}/logs"

export COSMOS_OUTPUT_FPS=24
export COSMOS_EXTRA_BIND_PATHS="${COSMOS_EXTRA_BIND_PATHS:-/datahdd}:${eval_root}"
export MUSA_VISIBLE_DEVICES="${gpu}"
export KMP_DUPLICATE_LIB_OK=TRUE

cd "${project}"
exec "${runner}" \
  examples/inference.py \
  -i "${input}" \
  -o "${output}" \
  --checkpoint-path "${weight}" \
  --experiment predict2_video2world_inference_2b_sam3d_action \
  --context-parallel-size 1 \
  --disable-guardrails \
  --keep-going \
  --skip-existing-output
