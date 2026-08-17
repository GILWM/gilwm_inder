#!/usr/bin/env bash
set -euo pipefail

node_index=${1:?usage: watch_hybrid_iter1000_then_bench50.sh NODE_INDEX}
if ((node_index < 0 || node_index > 3)); then
  echo "NODE_INDEX must be 0..3" >&2
  exit 2
fi

workspace=/datassd/morka/cosmos-sam3d-work
run_name=sam3d-v93-core15k-legacy4k-bs4-hybrid-r8iter5000-samiter1000-1k-norepa-20260816
training_root=${workspace}/training-outputs/${run_name}/cosmos_predict_v2p5_sam3d/video2world/${run_name}
checkpoint_dir=${training_root}/checkpoints/iter_000001000
latest_file=${training_root}/checkpoints/latest_checkpoint.txt
eval_root=/datassd/morka/cosmos-work/eval/sam3d_hybrid_r8iter5000_samiter1000_iter1000_bench50_ar93_640x480_121f_20260816
launcher=${workspace}/scripts/launch_sam3d_benchmark50_ar_workers.sh
watch_dir=${eval_root}/watch
marker=${watch_dir}/node_${node_index}.triggered
log=${watch_dir}/node_${node_index}.log

mkdir -p "${watch_dir}" "${eval_root}/logs" "${eval_root}/videos"
exec >>"${log}" 2>&1

echo "[$(date '+%F %T')] watcher_start node=${node_index}"
echo "training_root=${training_root}"
echo "checkpoint_dir=${checkpoint_dir}"
echo "eval_root=${eval_root}"

if [[ -f "${marker}" ]]; then
  echo "[$(date '+%F %T')] already_triggered marker=${marker}"
  exit 0
fi

while true; do
  latest=""
  if [[ -s "${latest_file}" ]]; then
    latest=$(tr -d '\r\n' <"${latest_file}")
  fi
  containers=$(docker ps -q | wc -l)
  if [[ "${latest}" == "iter_000001000" \
        && -s "${checkpoint_dir}/model/.metadata" \
        && -s "${checkpoint_dir}/model_ema_bf16.pt" \
        && "${containers}" -eq 0 ]]; then
    break
  fi
  echo "[$(date '+%F %T')] waiting latest=${latest:-none} containers=${containers}"
  sleep 15
done

echo "[$(date '+%F %T')] checkpoint_ready; launching local shards"
assignments=()
for gpu in 0 1 2 3 4 5 6 7; do
  shard=$((node_index * 8 + gpu))
  assignments+=("${shard}:${gpu}")
done

export WA_EVAL_ROOT="${eval_root}"
export WA_CHECKPOINT_DIR="${checkpoint_dir}"
"${launcher}" "${assignments[@]}"
printf '%s\n' "triggered_at=$(date '+%F %T')" "node_index=${node_index}" \
  "checkpoint_dir=${checkpoint_dir}" "eval_root=${eval_root}" >"${marker}"
sleep 10
echo "[$(date '+%F %T')] launched containers=$(docker ps -q | wc -l) marker=${marker}"
