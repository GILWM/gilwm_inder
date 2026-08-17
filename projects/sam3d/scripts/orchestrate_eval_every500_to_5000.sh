#!/usr/bin/env bash
set -euo pipefail

node_index=${1:?usage: orchestrate_eval_every500_to_5000.sh NODE_INDEX}
if ((node_index < 0 || node_index > 3)); then
  echo "NODE_INDEX must be 0..3" >&2
  exit 2
fi

workspace=/datassd/morka/cosmos-sam3d-work
run_name=sam3d-v93-core15k-legacy4k-bs4-hybrid-r8iter5000-samiter1000-1k-norepa-20260816
training_root=${workspace}/training-outputs/${run_name}/cosmos_predict_v2p5_sam3d/video2world/${run_name}
dataset=/datahdd/zhouhao/worldarena2_internal_benchmark50_320x240_121f_20260730
sweep_root=/datassd/morka/cosmos-work/eval/sam3d_hybrid_every500_bench50_ar93_640x480_121f_to5000_20260816
trainer=${workspace}/scripts/run_sam3d_v93_train.sh
runner=${workspace}/scripts/run_cosmos_sam3d_musa.sh
launcher=${workspace}/scripts/launch_sam3d_benchmark50_ar_workers.sh
prepare=${workspace}/scripts/prepare_sam3d_benchmark50_ar.py
audit=${workspace}/scripts/audit_sam3d_benchmark50_outputs.py
python=${workspace}/envs/cosmos-musa/bin/python
orchestrator_log=${sweep_root}/state/node_${node_index}_orchestrator.log

mkdir -p "${sweep_root}/state"
exec >>"${orchestrator_log}" 2>&1

echo "[$(date '+%F %T')] orchestrator_start node=${node_index}"
echo "training_root=${training_root}"
echo "sweep_root=${sweep_root}"

wait_for_local_idle() {
  while true; do
    local containers
    containers=$(docker ps -q | wc -l)
    if [[ "${containers}" -eq 0 ]]; then
      return 0
    fi
    echo "[$(date '+%F %T')] waiting_local_idle containers=${containers}"
    sleep 10
  done
}

run_eval() {
  local iter=$1
  local tag checkpoint_dir eval_root state_dir inputs ready_inputs ready_checkpoint valid marker
  tag=$(printf '%09d' "${iter}")
  checkpoint_dir=${training_root}/checkpoints/iter_${tag}
  eval_root=${sweep_root}/iter_${tag}
  state_dir=${eval_root}/state
  inputs=${eval_root}/inputs/inputs_640x480_121f_ar93_all.jsonl
  ready_inputs=${state_dir}/inputs.ready
  ready_checkpoint=${state_dir}/checkpoint.ready
  valid=${state_dir}/benchmark50.valid
  marker=${state_dir}/node_${node_index}.eval_launched

  mkdir -p "${state_dir}" "${eval_root}/logs" "${eval_root}/videos"
  echo "[$(date '+%F %T')] eval_stage_start iter=${iter}"

  while [[ ! -s "${checkpoint_dir}/model/.metadata" ]]; do
    echo "[$(date '+%F %T')] waiting_checkpoint iter=${iter}"
    sleep 15
  done
  wait_for_local_idle

  if ((node_index == 0)); then
    if [[ ! -f "${ready_inputs}" ]]; then
      "${python}" "${prepare}" "${dataset}" "${eval_root}/inputs" \
        --shards 32 \
        --name-prefix "hybrid_iter${iter}_ar93_640x480" \
        --num-steps 35 \
        >"${state_dir}/prepare.log" 2>&1
      test "$(wc -l <"${inputs}")" -eq 50
      test "$(find "${eval_root}/inputs" -maxdepth 1 -type f -name 'shard_*.jsonl' | wc -l)" -eq 32
      printf '%s\n' "prepared_at=$(date '+%F %T')" "samples=50" "shards=32" >"${ready_inputs}"
    fi
    if [[ ! -s "${checkpoint_dir}/model_ema_bf16.pt" ]]; then
      echo "[$(date '+%F %T')] converting_checkpoint iter=${iter}"
      "${runner}" scripts/convert_distcp_to_pt.py "${checkpoint_dir}/model" "${checkpoint_dir}" \
        >"${state_dir}/convert.log" 2>&1
    fi
    test -s "${checkpoint_dir}/model_ema_bf16.pt"
    printf '%s\n' "ready_at=$(date '+%F %T')" "checkpoint_dir=${checkpoint_dir}" >"${ready_checkpoint}"
  fi

  while [[ ! -f "${ready_inputs}" || ! -f "${ready_checkpoint}" ]]; do
    echo "[$(date '+%F %T')] waiting_eval_artifacts iter=${iter}"
    sleep 10
  done
  wait_for_local_idle

  if [[ ! -f "${marker}" ]]; then
    assignments=()
    for gpu in 0 1 2 3 4 5 6 7; do
      shard=$((node_index * 8 + gpu))
      assignments+=("${shard}:${gpu}")
    done
    export WA_EVAL_ROOT="${eval_root}"
    export WA_CHECKPOINT_DIR="${checkpoint_dir}"
    "${launcher}" "${assignments[@]}" >"${state_dir}/node_${node_index}_launcher.log" 2>&1
    printf '%s\n' "launched_at=$(date '+%F %T')" "node_index=${node_index}" >"${marker}"
    echo "[$(date '+%F %T')] eval_launched iter=${iter} node=${node_index}"
    sleep 15
  fi

  while [[ ! -f "${valid}" ]]; do
    mp4_count=$(find "${eval_root}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)
    containers=$(docker ps -q | wc -l)
    echo "[$(date '+%F %T')] eval_progress iter=${iter} mp4=${mp4_count}/50 local_containers=${containers}"
    if ((node_index == 0 && mp4_count == 50 && containers == 0)); then
      if "${python}" "${audit}" "${inputs}" "${eval_root}/videos" >"${state_dir}/audit.log" 2>&1; then
        printf '%s\n' "validated_at=$(date '+%F %T')" "videos=50" \
          "geometry=640x480x121@24fps" >"${valid}"
        echo "[$(date '+%F %T')] eval_valid iter=${iter}"
        break
      fi
      echo "[$(date '+%F %T')] eval_audit_failed iter=${iter}; retrying"
    fi
    sleep 15
  done

  wait_for_local_idle
  echo "[$(date '+%F %T')] eval_stage_complete iter=${iter}"
}

resume_to() {
  local target=$1
  local port marker done_marker log
  port=$((30000 + target / 500))
  marker=${sweep_root}/state/node_${node_index}_train_to_${target}.launched
  done_marker=${sweep_root}/state/node_${node_index}_train_to_${target}.done
  log=${sweep_root}/state/node_${node_index}_train_to_${target}.log

  if [[ -f "${done_marker}" ]]; then
    echo "[$(date '+%F %T')] training_segment_already_done target=${target}"
    return 0
  fi

  wait_for_local_idle
  printf '%s\n' "launched_at=$(date '+%F %T')" "node_index=${node_index}" \
    "target_iter=${target}" "save_iter=500" >"${marker}"
  echo "[$(date '+%F %T')] training_segment_start target=${target} port=${port}"

  unset SAM3D_INIT_CHECKPOINT SAM3D_COSMOS_OVERLAY_CHECKPOINT
  env \
    SAM3D_RUN_NAME="${run_name}" \
    MAX_ITER="${target}" \
    SAVE_ITER=500 \
    PER_DEVICE_BATCH_SIZE=4 \
    SAM3D_REPA_WEIGHT=0 \
    SAM3D_TEACHER_TOKENS_AS_CONDITION=false \
    SCHEDULER_CYCLE_LENGTH=5000 \
    SCHEDULER_WARMUP_STEPS=500 \
    MASTER_PORT="${port}" \
    "${trainer}" "${node_index}" train5000 \
    >>"${log}" 2>&1

  printf '%s\n' "completed_at=$(date '+%F %T')" "target_iter=${target}" >"${done_marker}"
  echo "[$(date '+%F %T')] training_segment_complete target=${target}"
}

# iter1000 evaluation is already complete.  iter1500 was saved before this
# orchestrator was installed, so backfill it after the active 1000->2000
# segment exits.  All later checkpoints are evaluated immediately at save.
run_eval 1500
run_eval 2000

for target in 2500 3000 3500 4000 4500 5000; do
  resume_to "${target}"
  run_eval "${target}"
done

echo "[$(date '+%F %T')] ORCHESTRATION_COMPLETE final_iter=5000"
