#!/usr/bin/env bash
set -euo pipefail

queue=${1:?usage: run_bdsys_eval_queue.sh QUEUE_INDEX_0_TO_3}
workspace=/bdsys_hdd/datassd/morka/cosmos-sam3d-work
launcher=${workspace}/scripts/launch_bdsys_eval_node.sh
checkpoint_root=/datassd/morka/cosmos-sam3d-work/training-outputs/sam3d-v93-complete106k-frommix121k-iter5000-lr1e5-bs4-32gpu-5k-20260831/cosmos_predict_v2p5_sam3d/video2world/sam3d-v93-complete106k-frommix121k-iter5000-lr1e5-bs4-32gpu-5k-20260831/checkpoints
state=/bdsys_hdd/datassd/morka/cosmos-work/eval/sam3d_complete106k_every500_balanced50_ar93_overlap01_640x480_121f_20260901/state

case "${queue}" in
  0) iterations=(500 2500 4500) ;;
  1) iterations=(1000 3000 5000) ;;
  2) iterations=(1500 3500) ;;
  3) iterations=(2000 4000) ;;
  *) echo "queue index must be 0..3" >&2; exit 2 ;;
esac

mkdir -p "${state}"
rm -f "${state}/queue_${queue}.DONE" "${state}/queue_${queue}.FAILED"
trap 'touch "${state}/queue_'"${queue}"'.FAILED"' ERR
for iteration in "${iterations[@]}"; do
  label=$(printf 'iter%05d' "${iteration}")
  checkpoint=${checkpoint_root}/$(printf 'iter_%09d' "${iteration}")
  echo "QUEUE_ITEM_START queue=${queue} label=${label} time=$(date --iso-8601=seconds)"
  "${launcher}" "${label}" "${checkpoint}"
  echo "QUEUE_ITEM_DONE queue=${queue} label=${label} time=$(date --iso-8601=seconds)"
done
touch "${state}/queue_${queue}.DONE"
rm -f "${state}/queue_${queue}.FAILED"
echo "QUEUE_DONE queue=${queue} time=$(date --iso-8601=seconds)"
