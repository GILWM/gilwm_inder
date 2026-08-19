#!/usr/bin/env bash
set -euo pipefail

# Current formal experiment: Core15K only, 93 frames, 1,000 optimizer steps.
# Action normalization is fixed from official WorldArena 1,000 + the actual
# >=93-frame Core15K training subset.

node_rank=${1:?usage: train_action_expert_core15k_1k.sh NODE_RANK}
data_root=${SAM3D_DATASET_ROOT:-/datahdd/mccxadmin/train_data}

export DATASET_MANIFEST_PATHS=${DATASET_MANIFEST_PATHS:-"['${data_root}/core15k_filter_75_455/manifest.jsonl']"}
export INCLUDED_BATCHES=${INCLUDED_BATCHES:-"['core15k']"}
export ACTION_NORM_PATH=${ACTION_NORM_PATH:-${data_root}/official1000_core15k_v93_action_norm_stats.npz}
export NUM_FRAMES=${NUM_FRAMES:-93}
export NNODES=${NNODES:-4}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
export MAX_ITER=${MAX_ITER:-1000}
export SAVE_ITER=${SAVE_ITER:-500}
export SAM3D_RUN_NAME=${SAM3D_RUN_NAME:-cosmos-action-expert-core15k-v93-bs${PER_DEVICE_BATCH_SIZE}-1k}

# The current hosts do not expose the same RDMA-device name for ens11np0
# (wx25: mlx5_0; wx26-wx28: mlx5_2). Resolve it on every node instead of
# assigning one global MCCL_IB_HCA value, which corrupts cross-node rings.
export MCCL_SOCKET_IFNAME=${MCCL_SOCKET_IFNAME:-ens11np0}
if [[ -z "${MCCL_IB_HCA:-}" ]] && command -v ibdev2netdev >/dev/null 2>&1; then
  MCCL_IB_HCA=$(ibdev2netdev | awk -v iface="${MCCL_SOCKET_IFNAME}" '$5 == iface { print $1; exit }')
  export MCCL_IB_HCA
fi
if [[ -z "${MCCL_IB_HCA:-}" ]]; then
  echo "Could not resolve the RDMA device for ${MCCL_SOCKET_IFNAME}" >&2
  exit 2
fi

export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-20971520}
export MCCL_MIN_NRINGS=${MCCL_MIN_NRINGS:-1}
export MCCL_MAX_NRINGS=${MCCL_MAX_NRINGS:-1}

exec "$(dirname "$0")/train_action_expert_filter_75_455.sh" "${node_rank}"
