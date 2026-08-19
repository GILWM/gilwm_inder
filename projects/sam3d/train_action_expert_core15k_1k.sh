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

exec "$(dirname "$0")/train_action_expert_filter_75_455.sh" "${node_rank}"
