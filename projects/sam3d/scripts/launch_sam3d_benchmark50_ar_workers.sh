#!/usr/bin/env bash
set -euo pipefail

worker=${COSMOS_BENCH_WORKER:-/datassd/morka/cosmos-sam3d-work/scripts/run_sam3d_benchmark50_ar_worker.sh}
eval_root=${WA_EVAL_ROOT:-/datassd/morka/cosmos-work/eval/sam3d_iter1000_bench50_ar93_640x480_121f_20260815}

if (($# == 0)); then
  echo "usage: $0 SHARD:GPU [SHARD:GPU ...]" >&2
  exit 2
fi

mkdir -p "${eval_root}/logs"

for assignment in "$@"; do
  shard=${assignment%%:*}
  gpu=${assignment##*:}
  log=${eval_root}/logs/launcher_$(printf '%02d' "${shard}").log
  nohup "${worker}" "${shard}" "${gpu}" >"${log}" 2>&1 < /dev/null &
  echo "started shard=${shard} gpu=${gpu} pid=$!"
done
