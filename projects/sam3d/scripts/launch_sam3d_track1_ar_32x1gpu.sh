#!/usr/bin/env bash
set -euo pipefail

host_index=${1:?usage: launch_sam3d_track1_ar_32x1gpu.sh HOST_INDEX HOST_COUNT}
host_count=${2:?usage: launch_sam3d_track1_ar_32x1gpu.sh HOST_INDEX HOST_COUNT}

workspace=${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}
project=${COSMOS_PROJECT:-${workspace}/cosmos-predict2.5-sam3d-musa}
runner=${COSMOS_RUNNER:-${workspace}/scripts/run_cosmos_sam3d_musa.sh}
eval_root=${WA_EVAL_ROOT:?WA_EVAL_ROOT must point to the prepared evaluation directory}
checkpoint_dir=${WA_CHECKPOINT_DIR:-${workspace}/training-outputs/sam3d-v93-core15k-legacy4k-bs4-5k-norepa-20260814/cosmos_predict_v2p5_sam3d/video2world/sam3d-v93-core15k-legacy4k-bs4-5k-norepa-20260814/checkpoints/iter_000001000}
weight=${checkpoint_dir}/model_ema_bf16.pt
output=${eval_root}/videos
logs=${eval_root}/logs
status=${eval_root}/status

if ((host_index < 0 || host_index >= host_count)); then
  echo "HOST_INDEX must be in [0, HOST_COUNT)" >&2
  exit 2
fi
test -s "${weight}"
mkdir -p "${output}" "${logs}" "${status}"

export COSMOS_OUTPUT_FPS=24
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KMP_DUPLICATE_LIB_OK=TRUE

cd "${project}"
pids=()
shards=()
for local_gpu in 0 1 2 3 4 5 6 7; do
  # Interleave shards across hosts: every host receives exactly 250 samples.
  shard=$((local_gpu * host_count + host_index))
  input=${eval_root}/inputs/shard_$(printf '%02d' "${shard}").jsonl
  log=${logs}/worker_$(printf '%02d' "${shard}")_$(hostname).log
  test -s "${input}"
  env MUSA_VISIBLE_DEVICES=${local_gpu} "${runner}" \
    examples/inference.py \
    -i "${input}" \
    -o "${output}" \
    --checkpoint-path "${weight}" \
    --experiment predict2_video2world_training_2b_sam3d_smoke \
    --context-parallel-size 1 \
    --disable-guardrails \
    --keep-going \
    --skip-existing-output \
    >"${log}" 2>&1 &
  pids+=("$!")
  shards+=("${shard}")
  echo "started host=$(hostname) gpu=${local_gpu} shard=${shard} pid=$! log=${log}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo 0 >"${status}/shard_$(printf '%02d' "${shards[$index]}").exit"
  else
    code=$?
    echo "${code}" >"${status}/shard_$(printf '%02d' "${shards[$index]}").exit"
    failed=1
  fi
done
echo "${failed}" >"${status}/host_${host_index}.exit"
exit "${failed}"
