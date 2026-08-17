#!/usr/bin/env bash
set -euo pipefail

node_index=${1:?usage: watch_bench50_then_resume_to_2000.sh NODE_INDEX}
if ((node_index < 0 || node_index > 3)); then
  echo "NODE_INDEX must be 0..3" >&2
  exit 2
fi

workspace=/datassd/morka/cosmos-sam3d-work
run_name=sam3d-v93-core15k-legacy4k-bs4-hybrid-r8iter5000-samiter1000-1k-norepa-20260816
training_root=${workspace}/training-outputs/${run_name}/cosmos_predict_v2p5_sam3d/video2world/${run_name}
resume_checkpoint=${training_root}/checkpoints/iter_000001000
eval_root=/datassd/morka/cosmos-work/eval/sam3d_hybrid_r8iter5000_samiter1000_iter1000_bench50_ar93_640x480_121f_20260816
inputs=${eval_root}/inputs/inputs_640x480_121f_ar93_all.jsonl
videos=${eval_root}/videos
watch_dir=${eval_root}/resume-watch
ready_marker=${watch_dir}/benchmark50.valid
trigger_marker=${watch_dir}/node_${node_index}.training_launched
log=${watch_dir}/node_${node_index}.log
audit_log=${watch_dir}/benchmark50_audit.log
training_log=${watch_dir}/node_${node_index}_resume_training.log
trainer=${workspace}/scripts/run_sam3d_v93_train.sh
python=${workspace}/envs/cosmos-musa/bin/python
audit=${workspace}/scripts/audit_sam3d_benchmark50_outputs.py

mkdir -p "${watch_dir}"
exec >>"${log}" 2>&1

echo "[$(date '+%F %T')] resume_watcher_start node=${node_index}"
echo "eval_root=${eval_root}"
echo "resume_checkpoint=${resume_checkpoint}"

if [[ -f "${trigger_marker}" ]]; then
  echo "[$(date '+%F %T')] already_launched marker=${trigger_marker}"
  exit 0
fi

test -s "${inputs}"
for part in model optim scheduler trainer; do
  test -s "${resume_checkpoint}/${part}/.metadata"
done

while [[ ! -f "${ready_marker}" ]]; do
  mp4_count=$(find "${videos}" -maxdepth 1 -type f -name '*.mp4' | wc -l)
  containers=$(docker ps -q | wc -l)
  echo "[$(date '+%F %T')] waiting_for_eval mp4=${mp4_count}/50 local_containers=${containers}"
  if ((node_index == 0 && mp4_count == 50 && containers == 0)); then
    if "${python}" "${audit}" "${inputs}" "${videos}" >"${audit_log}" 2>&1; then
      printf '%s\n' "validated_at=$(date '+%F %T')" "videos=50" \
        "geometry=640x480x121@24fps" >"${ready_marker}"
      echo "[$(date '+%F %T')] benchmark_valid marker=${ready_marker}"
      break
    fi
    echo "[$(date '+%F %T')] audit_not_ready; retrying"
  fi
  sleep 15
done

while true; do
  containers=$(docker ps -q | wc -l)
  if [[ "${containers}" -eq 0 ]]; then
    break
  fi
  echo "[$(date '+%F %T')] benchmark_valid_waiting_local_cleanup containers=${containers}"
  sleep 5
done

printf '%s\n' "launched_at=$(date '+%F %T')" "node_index=${node_index}" \
  "resume_checkpoint=${resume_checkpoint}" "max_iter=2000" "save_iter=500" >"${trigger_marker}"
echo "[$(date '+%F %T')] launching_resume node=${node_index} max_iter=2000 save_iter=500"

unset SAM3D_INIT_CHECKPOINT SAM3D_COSMOS_OVERLAY_CHECKPOINT
exec env \
  SAM3D_RUN_NAME="${run_name}" \
  MAX_ITER=2000 \
  SAVE_ITER=500 \
  PER_DEVICE_BATCH_SIZE=4 \
  SAM3D_REPA_WEIGHT=0 \
  SAM3D_TEACHER_TOKENS_AS_CONDITION=false \
  SCHEDULER_CYCLE_LENGTH=5000 \
  SCHEDULER_WARMUP_STEPS=500 \
  MASTER_PORT=29938 \
  "${trainer}" "${node_index}" train5000 \
  >>"${training_log}" 2>&1
