#!/usr/bin/env bash
set -euo pipefail

host_index=${1:?usage: launch_sam3d_track1_dual_pass_32gpu.sh HOST_INDEX HOST_COUNT}
host_count=${2:?usage: launch_sam3d_track1_dual_pass_32gpu.sh HOST_INDEX HOST_COUNT}

eval_a=${WA_EVAL_ROOT_A:?WA_EVAL_ROOT_A is required}
eval_b=${WA_EVAL_ROOT_B:?WA_EVAL_ROOT_B is required}
workspace=${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}
project=${COSMOS_PROJECT:-${workspace}/cosmos-predict2.5-sam3d-musa}
runner=${COSMOS_RUNNER:-${workspace}/scripts/run_cosmos_sam3d_musa.sh}
checkpoint_dir=${WA_CHECKPOINT_DIR:-${workspace}/training-outputs/sam3d-v93-core15k-legacy4k-bs4-5k-norepa-20260814/cosmos_predict_v2p5_sam3d/video2world/sam3d-v93-core15k-legacy4k-bs4-5k-norepa-20260814/checkpoints/iter_000001000}
weight=${checkpoint_dir}/model_ema_bf16.pt

if ((host_index < 0 || host_index >= host_count)); then
  echo "HOST_INDEX must be in [0, HOST_COUNT)" >&2
  exit 2
fi
test -s "${weight}"
for root in "${eval_a}" "${eval_b}"; do
  mkdir -p "${root}/videos" "${root}/logs" "${root}/status"
done

export COSMOS_OUTPUT_FPS=24
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE

cd "${project}"
pids=()
roots=()
shards=()

launch_worker() {
  local root=$1
  local local_gpu=$2
  local shard=$3
  local label=$4
  local input=${root}/inputs/shard_$(printf '%02d' "${shard}").jsonl
  local log=${root}/logs/worker_$(printf '%02d' "${shard}")_$(hostname).log
  test -s "${input}"
  env MUSA_VISIBLE_DEVICES=${local_gpu} "${runner}" \
    examples/inference.py \
    -i "${input}" \
    -o "${root}/videos" \
    --checkpoint-path "${weight}" \
    --experiment predict2_video2world_training_2b_sam3d_smoke \
    --context-parallel-size 1 \
    --disable-guardrails \
    --keep-going \
    --skip-existing-output \
    >"${log}" 2>&1 &
  pids+=("$!")
  roots+=("${root}")
  shards+=("${shard}")
  echo "started pass=${label} host=$(hostname) gpu=${local_gpu} shard=${shard} pid=$!"
}

# Four workers per pass on every host: 16 GPUs for A and 16 GPUs for B.
for slot in 0 1 2 3; do
  shard=$((slot * host_count + host_index))
  launch_worker "${eval_a}" "${slot}" "${shard}" A
  launch_worker "${eval_b}" "$((slot + 4))" "${shard}" B
done

failed=0
for index in "${!pids[@]}"; do
  root=${roots[$index]}
  shard=${shards[$index]}
  if wait "${pids[$index]}"; then
    echo 0 >"${root}/status/shard_$(printf '%02d' "${shard}").exit"
  else
    code=$?
    echo "${code}" >"${root}/status/shard_$(printf '%02d' "${shard}").exit"
    failed=1
  fi
done
echo "${failed}" >"${eval_a}/status/host_${host_index}.exit"
echo "${failed}" >"${eval_b}/status/host_${host_index}.exit"
exit "${failed}"
