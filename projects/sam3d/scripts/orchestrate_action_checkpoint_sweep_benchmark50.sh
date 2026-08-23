#!/usr/bin/env bash
set -euo pipefail

node_index=${1:?usage: orchestrate_action_checkpoint_sweep_benchmark50.sh NODE_INDEX}
if ((node_index < 0 || node_index > 3)); then
  echo "NODE_INDEX must be 0..3" >&2
  exit 2
fi

workspace=${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}
project=${COSMOS_PROJECT:-${workspace}/action-alignment-audit-code-20260820}
run_name=action-core15k-robotwin2-14d-v93-continuous-bs4-init4000-5k-20260821-r1
training_root=${workspace}/training-outputs/${run_name}/cosmos_predict_v2p5_sam3d/video2world/${run_name}
benchmark_root=/datahdd/zhouhao/worldarena-evaluator/benchmarks/internal/robotwin2_clean50_640x480_121f
action_norm=/datahdd/mccxadmin/train_data/official1000_core15k_v93_action_norm_stats.npz
sweep_root=/datassd/morka/cosmos-work/eval/action-continuous93-init4000-every500-bench50-fixed121-nativeh5-20260823

runner=${project}/projects/sam3d/scripts/run_cosmos_sam3d_musa.sh
prepare=${project}/projects/sam3d/scripts/prepare_action_benchmark50_ar.py
worker=${project}/projects/sam3d/scripts/run_action_benchmark50_worker.sh
audit=${project}/projects/sam3d/scripts/audit_sam3d_benchmark50_outputs.py
python=${workspace}/envs/cosmos-musa/bin/python
state_root=${sweep_root}/state
setup_complete=${state_root}/node_${node_index}.setup.complete
setup_failed=${state_root}/node_${node_index}.setup.failed
iterations=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000)
case ${node_index} in
  0) setup_iterations=(500 1000 4500 5000) ;;
  1) setup_iterations=(1500 2000) ;;
  2) setup_iterations=(2500 3000) ;;
  3) setup_iterations=(3500 4000) ;;
esac

export COSMOS_SAM3D_WORKSPACE="${workspace}"
export COSMOS_PROJECT="${project}"
export COSMOS_MUSA_IMAGE="${COSMOS_MUSA_IMAGE:-10.200.53.208/ci/nanhu-computing-framework:v2.1.6-rc1}"

mkdir -p "${state_root}"
exec >>"${state_root}/node_${node_index}_orchestrator.log" 2>&1

timestamp() { date '+%F %T'; }

checkpoint_dir_for() {
  printf '%s/checkpoints/iter_%09d' "${training_root}" "$1"
}

eval_root_for() {
  local iter=$1 mode=$2
  printf '%s/iter_%09d/%s' "${sweep_root}" "${iter}" "${mode}"
}

wait_for_ready() {
  local marker=$1
  while [[ ! -f "${marker}" ]]; do
    if [[ -s "${setup_failed}" ]]; then
      echo "[$(timestamp)] setup_failed while waiting for ${marker}"
      cat "${setup_failed}"
      return 1
    fi
    sleep 10
  done
}

prepare_mode() {
  local iter=$1 mode=$2 frame_mode=$3 prefix checkpoint_dir eval_root ready
  checkpoint_dir=$(checkpoint_dir_for "${iter}")
  eval_root=$(eval_root_for "${iter}" "${mode}")
  ready=${eval_root}/state/inputs.ready
  mkdir -p "${eval_root}/inputs" "${eval_root}/videos" "${eval_root}/logs" "${eval_root}/state"
  if [[ ! -f "${ready}" ]]; then
    "${python}" "${prepare}" "${benchmark_root}" "${eval_root}/inputs" \
      --action-norm-path "${action_norm}" \
      --shards 2 \
      --name-prefix "continuous_iter$(printf '%04d' "${iter}")_${mode}" \
      --num-steps 35 \
      --guidance 7 \
      --frame-count-mode "${frame_mode}" \
      >"${eval_root}/state/prepare.log" 2>&1
    test "$(wc -l <"${eval_root}/inputs/all.jsonl")" -eq 50
    test "$(find "${eval_root}/inputs" -maxdepth 1 -type f -name 'shard_*.jsonl' | wc -l)" -eq 2
    printf 'prepared_at=%s\nmode=%s\nsamples=50\nshards=2\n' \
      "$(timestamp)" "${frame_mode}" >"${ready}"
  fi
}

run_smoke() {
  local iter=500 checkpoint_dir fixed_root smoke_root
  checkpoint_dir=$(checkpoint_dir_for "${iter}")
  fixed_root=$(eval_root_for "${iter}" fixed121)
  smoke_root=${sweep_root}/smoke_iter_000000500_fixed121
  if [[ -f "${state_root}/smoke.valid" ]]; then
    return 0
  fi
  mkdir -p "${smoke_root}/inputs" "${smoke_root}/videos" "${smoke_root}/logs"
  head -n 1 "${fixed_root}/inputs/all.jsonl" >"${smoke_root}/inputs/shard_00.jsonl"
  echo "[$(timestamp)] smoke_start checkpoint=${checkpoint_dir}"
  "${worker}" "${smoke_root}" "${checkpoint_dir}" 0 0 \
    >"${state_root}/smoke.log" 2>&1
  test "$(find "${smoke_root}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 1
  printf 'validated_at=%s\ncheckpoint=%s\n' "$(timestamp)" "${checkpoint_dir}" \
    >"${state_root}/smoke.valid"
  echo "[$(timestamp)] smoke_valid"
}

setup_all() {
  rm -f "${setup_failed}"
  for iter in "${setup_iterations[@]}"; do
    local checkpoint_dir checkpoint_ready
    checkpoint_dir=$(checkpoint_dir_for "${iter}")
    checkpoint_ready=${sweep_root}/iter_$(printf '%09d' "${iter}")/state/checkpoint.ready
    mkdir -p "$(dirname "${checkpoint_ready}")"
    echo "[$(timestamp)] setup_start iter=${iter}"
    test -s "${checkpoint_dir}/model/.metadata"
    prepare_mode "${iter}" fixed121 fixed
    prepare_mode "${iter}" hdf5_native action
    if [[ ! -s "${checkpoint_dir}/model_ema_bf16.pt" ]]; then
      echo "[$(timestamp)] convert_start iter=${iter}"
      "${runner}" scripts/convert_distcp_to_pt.py "${checkpoint_dir}/model" "${checkpoint_dir}" \
        >"${state_root}/convert_iter_$(printf '%09d' "${iter}").log" 2>&1
      echo "[$(timestamp)] convert_done iter=${iter}"
    fi
    test -s "${checkpoint_dir}/model_ema_bf16.pt"
    if ((node_index == 0 && iter == 500)); then
      run_smoke
    fi
    printf 'ready_at=%s\ncheckpoint_dir=%s\n' "$(timestamp)" "${checkpoint_dir}" \
      >"${checkpoint_ready}"
    echo "[$(timestamp)] setup_ready iter=${iter}"
  done
  printf 'completed_at=%s\ncheckpoints=%s\nexperiments=%s\n' \
    "$(timestamp)" "${#setup_iterations[@]}" "$(( ${#setup_iterations[@]} * 2 ))" \
    >"${setup_complete}"
}

run_task() {
  local task=$1 gpu=$2 group rem iter mode shard checkpoint_dir eval_root ready task_name log status
  group=$((task / 4))
  rem=$((task % 4))
  iter=${iterations[${group}]}
  if ((rem < 2)); then
    mode=fixed121
    shard=${rem}
  else
    mode=hdf5_native
    shard=$((rem - 2))
  fi
  checkpoint_dir=$(checkpoint_dir_for "${iter}")
  eval_root=$(eval_root_for "${iter}" "${mode}")
  ready=${sweep_root}/iter_$(printf '%09d' "${iter}")/state/checkpoint.ready
  task_name=iter_$(printf '%09d' "${iter}")_${mode}_shard_$(printf '%02d' "${shard}")
  log=${state_root}/${task_name}.log
  status=${state_root}/${task_name}.status
  wait_for_ready "${ready}"
  echo "[$(timestamp)] task_start task=${task} gpu=${gpu} name=${task_name}"
  if "${worker}" "${eval_root}" "${checkpoint_dir}" "${shard}" "${gpu}" >"${log}" 2>&1; then
    printf 'exit_code=0\ncompleted_at=%s\nnode=%s\ngpu=%s\n' \
      "$(timestamp)" "${node_index}" "${gpu}" >"${status}"
    echo "[$(timestamp)] task_done task=${task} gpu=${gpu} name=${task_name}"
  else
    local rc=$?
    printf 'exit_code=%s\nfailed_at=%s\nnode=%s\ngpu=%s\n' \
      "${rc}" "$(timestamp)" "${node_index}" "${gpu}" >"${status}"
    echo "[$(timestamp)] task_failed rc=${rc} task=${task} gpu=${gpu} name=${task_name}"
    return "${rc}"
  fi
}

run_slot() {
  local gpu=$1 task=$((node_index * 8 + gpu))
  while ((task < 40)); do
    run_task "${task}" "${gpu}"
    task=$((task + 32))
  done
}

audit_all() {
  local overall=0 iter mode eval_root audit_log
  for iter in "${setup_iterations[@]}"; do
    for mode in fixed121 hdf5_native; do
      eval_root=$(eval_root_for "${iter}" "${mode}")
      audit_log=${eval_root}/state/audit.log
      if "${python}" "${audit}" "${eval_root}/inputs/all.jsonl" "${eval_root}/videos" \
        >"${audit_log}" 2>&1; then
        printf 'validated_at=%s\n' "$(timestamp)" >"${eval_root}/state/benchmark50.valid"
      else
        printf 'failed_at=%s\n' "$(timestamp)" >"${eval_root}/state/benchmark50.invalid"
        overall=1
      fi
    done
  done
  return "${overall}"
}

echo "[$(timestamp)] orchestrator_start node=${node_index}"
echo "training_root=${training_root}"
echo "benchmark_root=${benchmark_root}"
echo "sweep_root=${sweep_root}"

setup_pid=
if [[ ! -f "${setup_complete}" ]]; then
  (
    trap '
      rc=$?
      if ((rc != 0)); then
        printf "exit_code=%s\nfailed_at=%s\n" "${rc}" "$(timestamp)" >"${setup_failed}"
      fi
    ' EXIT
    setup_all
  ) &
  setup_pid=$!
fi

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_slot "${gpu}" &
  pids+=("$!")
done

node_rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || node_rc=1
done
if [[ -n "${setup_pid}" ]]; then
  wait "${setup_pid}" || node_rc=1
fi
printf 'completed_at=%s\nnode=%s\nexit_code=%s\n' "$(timestamp)" "${node_index}" "${node_rc}" \
  >"${state_root}/node_${node_index}.workers_done"

if audit_all; then
  printf 'completed_at=%s\nnode=%s\ncheckpoints=%s\nexperiments=%s\nvideos=%s\n' \
    "$(timestamp)" "${node_index}" "${#setup_iterations[@]}" \
    "$(( ${#setup_iterations[@]} * 2 ))" "$(( ${#setup_iterations[@]} * 100 ))" \
    >"${state_root}/node_${node_index}.SWEEP_COMPLETE"
  echo "[$(timestamp)] NODE_SWEEP_COMPLETE node=${node_index} videos=$(( ${#setup_iterations[@]} * 100 ))"
else
  node_rc=1
  printf 'failed_at=%s\nnode=%s\nreason=audit_failed\n' \
    "$(timestamp)" "${node_index}" >"${state_root}/node_${node_index}.SWEEP_INCOMPLETE"
  echo "[$(timestamp)] NODE_SWEEP_INCOMPLETE node=${node_index} audit_failed"
fi

exit "${node_rc}"
