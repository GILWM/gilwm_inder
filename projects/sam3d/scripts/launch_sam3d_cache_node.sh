#!/usr/bin/env bash
set -euo pipefail

node_rank=${1:?usage: launch_sam3d_cache_node.sh NODE_RANK}
if (( node_rank < 0 || node_rank > 3 )); then
  echo "NODE_RANK must be 0..3" >&2
  exit 2
fi

workspace=/datassd/morka/cosmos-sam3d-work
manifest=/datahdd/mccxadmin/cosmos-sam3d-manifests/core15k_legacy4k.jsonl
cache_root=/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93
log_root="${workspace}/logs/cache-v93"
num_shards=32

test -s "$manifest"
test -s "${workspace}/weights/sam3/sam3.pt"
mkdir -p "$cache_root" "$log_root"

for local_rank in $(seq 0 7); do
  shard=$((node_rank * 8 + local_rank))
  log="${log_root}/shard-${shard}.log"
  pidfile="${log_root}/shard-${shard}.pid"
  if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "shard $shard already running as PID $(cat "$pidfile")"
    continue
  fi
  MUSA_VISIBLE_DEVICES="$local_rank" nohup \
    "${workspace}/scripts/run_condition_preprocess_musa.sh" \
    "${workspace}/scripts/prepare_sam3d_conditions.py" \
      --dataset-root /datahdd/mccxadmin/train_data \
      --manifest "$manifest" \
      --cache-root "$cache_root" \
      --sam3-checkpoint "${workspace}/weights/sam3/sam3.pt" \
      --dino-repo "${workspace}/third_party/dinov2-official" \
      --device musa:0 \
      --teacher-frames 8 \
      --mask-instances 8 \
      --minimum-frames 93 \
      --num-shards "$num_shards" \
      --shard-index "$shard" \
      >"$log" 2>&1 </dev/null &
  echo $! > "$pidfile"
  echo "started shard=$shard local_gpu=$local_rank pid=$! log=$log"
done
