#!/usr/bin/env bash
set -euo pipefail

label=${1:?usage: launch_bdsys_eval_node.sh LABEL CHECKPOINT_DIR}
checkpoint_dir=${2:?usage: launch_bdsys_eval_node.sh LABEL CHECKPOINT_DIR}
host_workspace=/bdsys_hdd/datassd/morka/cosmos-sam3d-work
container_workspace=/datassd/morka/cosmos-sam3d-work
project=${container_workspace}/cosmos-predict2.5-sam3d-noaction-20260827
runner=${host_workspace}/scripts/run_cosmos_sam3d_musa_bdsys.sh
worker=${host_workspace}/scripts/run_bdsys_benchmark_worker.sh
convert=${project}/scripts/convert_distcp_to_pt.py
container_eval_base=/datassd/morka/cosmos-work/eval/sam3d_complete106k_every500_balanced50_ar93_overlap01_640x480_121f_20260901
host_eval_base=/bdsys_hdd${container_eval_base}
host_checkpoint=/bdsys_hdd${checkpoint_dir}
node_state=${host_eval_base}/state/${label}
mkdir -p "${node_state}"
rm -f "${node_state}/DONE" "${node_state}/FAILED"
trap 'touch "${node_state}/FAILED"' ERR

if [[ ! -s "${host_checkpoint}/model_ema_bf16.pt" ]]; then
  echo "CONVERT_START label=${label} time=$(date --iso-8601=seconds)"
  MUSA_VISIBLE_DEVICES=0 COSMOS_PROJECT="${project}" SAM3D_CONDITION_PROOF=0 \
    "${runner}" "${convert}" "${checkpoint_dir}/model" "${checkpoint_dir}" \
    >"${node_state}/convert.log" 2>&1
  test -s "${host_checkpoint}/model_ema_bf16.pt"
  echo "CONVERT_DONE label=${label} bytes=$(stat -c %s "${host_checkpoint}/model_ema_bf16.pt")"
fi

pids=()
for variant in with_condition no_condition; do
  if [[ "${variant}" == with_condition ]]; then offset=0; else offset=4; fi
  root=${container_eval_base}/${label}/${variant}
  host_root=/bdsys_hdd${root}
  rm -f "${host_root}/logs"/worker_*.log
  for shard in 0 1 2 3; do
    gpu=$((offset + shard))
    WA_EVAL_ROOT="${root}" WA_CHECKPOINT_DIR="${checkpoint_dir}" \
      COSMOS_PROJECT="${project}" COSMOS_RUNNER="${runner}" \
      nohup "${worker}" "${shard}" "${gpu}" \
      >"${host_root}/logs/launcher_$(printf '%02d' "${shard}").log" 2>&1 < /dev/null &
    pids+=("$!")
  done
done
printf '%s\n' "${pids[@]}" >"${node_state}/pids"

# Fail closed: every process must prove the expected network-side behavior on
# its first denoising call before this checkpoint is allowed to continue.
deadline=$((SECONDS + 1200))
while (( SECONDS < deadline )); do
  with_proofs=$(grep -l 'SAM3D_CONDITION_PROOF effective=true active_modalities=mask,mask_meta,geometry,shape,pose teacher_as_condition=false' \
    "${host_eval_base}/${label}/with_condition/logs"/worker_*.log 2>/dev/null | wc -l || true)
  without_proofs=$(grep -l 'SAM3D_CONDITION_PROOF effective=false active_modalities=none teacher_as_condition=false' \
    "${host_eval_base}/${label}/no_condition/logs"/worker_*.log 2>/dev/null | wc -l || true)
  echo "PROOF_WAIT label=${label} with=${with_proofs}/4 without=${without_proofs}/4 time=$(date --iso-8601=seconds)"
  if [[ "${with_proofs}" -eq 4 && "${without_proofs}" -eq 4 ]]; then
    break
  fi
  dead=0
  for pid in "${pids[@]}"; do kill -0 "${pid}" 2>/dev/null || dead=$((dead + 1)); done
  if (( dead > 0 )); then
    echo "A worker exited before producing a valid condition proof" >&2
    kill "${pids[@]}" 2>/dev/null || true
    exit 1
  fi
  sleep 15
done
if [[ "${with_proofs:-0}" -ne 4 || "${without_proofs:-0}" -ne 4 ]]; then
  echo "Condition proof timed out" >&2
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
fi
touch "${node_state}/CONDITION_PROOF_VALID"

for pid in "${pids[@]}"; do wait "${pid}"; done

for variant in with_condition no_condition; do
  host_root=${host_eval_base}/${label}/${variant}
  count=$(find "${host_root}/videos" -maxdepth 1 -name '*.mp4' -type f -size +0c | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "${label}/${variant}: expected 50 videos, got ${count}" >&2
    exit 1
  fi
  python3 "${host_workspace}/scripts/audit_sam3d_benchmark50_outputs.py" \
    "${host_root}/inputs/all.jsonl" "${host_root}/videos" >"${host_root}/state/audit.log" 2>&1
  touch "${host_root}/state/VALID"
done
touch "${node_state}/DONE"
rm -f "${node_state}/FAILED"
echo "EVAL_DONE label=${label} time=$(date --iso-8601=seconds)"
