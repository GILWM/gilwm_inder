#!/usr/bin/env bash
set -euo pipefail

# Cosmos-native FlowWAM-style inverse-dynamics experiment.
# The two manifests are combined in memory; videos/actions remain in place.

node_rank=${1:?usage: train_action_expert_filter_75_455.sh NODE_RANK}
data_root=${SAM3D_DATASET_ROOT:-/datahdd/mccxadmin/train_data}

export SAM3D_DATASET_ROOT="${data_root}"
export DATASET_MANIFEST_PATHS=${DATASET_MANIFEST_PATHS:-"['${data_root}/core15k_filter_75_455/manifest.jsonl','${data_root}/robotwin2_filter_75_455/manifest.jsonl']"}
export INCLUDED_BATCHES=${INCLUDED_BATCHES:-"['core15k','robotwin2']"}
export NUM_FRAMES=${NUM_FRAMES:-93}
# Both filtered views are symlink farms whose video targets live elsewhere
# under /datahdd. The container must see those target paths too.
export COSMOS_EXTRA_BIND_PATHS=${COSMOS_EXTRA_BIND_PATHS:-/datahdd}

# These filtered views already contain action HDF5 paths in each manifest.
# Missing SAM/SAM3D caches become zero sidecars instead of dropping samples.
export SAM3D_REQUIRED=${SAM3D_REQUIRED:-false}
export ACTION_NORM_PATH=${ACTION_NORM_PATH:-${data_root}/filter_75_455_v93_action_norm_stats.npz}
export ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}
export ACTION_ALIGNMENT_WEIGHT=${ACTION_ALIGNMENT_WEIGHT:-0.1}
export ACTION_ARCHITECTURE=cosmos_action_expert
export ACTION_HIDDEN_DIM=${ACTION_HIDDEN_DIM:-512}
export ACTION_NUM_LAYERS=${ACTION_NUM_LAYERS:-6}
export ACTION_NUM_HEADS=${ACTION_NUM_HEADS:-8}
export ACTION_FFN_MULTIPLIER=${ACTION_FFN_MULTIPLIER:-4}
export ACTION_POOL_GRID=${ACTION_POOL_GRID:-2}
export ACTION_FEATURE_LAYERS=${ACTION_FEATURE_LAYERS:-'[3,7,11,15,19,23]'}
export ACTION_FEATURE_LAYER=${ACTION_FEATURE_LAYER:-23}

if [[ ! -s "${ACTION_NORM_PATH}" ]]; then
  cat >&2 <<EOF
Missing ${ACTION_NORM_PATH}
Build the exact >=${NUM_FRAMES}-frame two-dataset statistics first:
python projects/sam3d/scripts/audit_worldarena_action_hdf5.py \
  ${data_root}/core15k_filter_75_455 \
  ${data_root}/robotwin2_filter_75_455 \
  --min-video-frames ${NUM_FRAMES} \
  --skip-schema-audit --progress-every 1000 \
  --write-norm ${ACTION_NORM_PATH} \
  --json-report ${ACTION_NORM_PATH%.npz}.json
EOF
  exit 2
fi

exec "$(dirname "$0")/train_v93.sh" "${node_rank}"
