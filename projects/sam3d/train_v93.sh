#!/usr/bin/env bash
set -euo pipefail

node_rank=${1:?usage: train_v93.sh NODE_RANK}
repo_root=${COSMOS_PROJECT:-$(git rev-parse --show-toplevel)}
workspace=${COSMOS_SAM3D_WORKSPACE:?set COSMOS_SAM3D_WORKSPACE}
dataset_root=${SAM3D_DATASET_ROOT:?set SAM3D_DATASET_ROOT}
cache_root=${SAM3D_CACHE_ROOT:?set SAM3D_CACHE_ROOT}
objects_root=${SAM3D_OBJECTS_ROOT:-}
base_checkpoint=${SAM3D_BASE_CHECKPOINT:?set SAM3D_BASE_CHECKPOINT}

nnodes=${NNODES:-4}
master_addr=${MASTER_ADDR:?set MASTER_ADDR}
master_port=${MASTER_PORT:-29932}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}
max_iter=${MAX_ITER:-5000}
save_iter=${SAVE_ITER:-500}
run_name=${SAM3D_RUN_NAME:-sam3d-v93-bs${per_device_batch_size}-5k}
learning_rate=${LEARNING_RATE:-3.0e-5}
repa_weight=${SAM3D_REPA_WEIGHT:-0}
repa_mode=${SAM3D_REPA_MODE:-projected_cosine}
repa_layer=${SAM3D_REPA_LAYER:-7}
repa_temporal_weight=${SAM3D_REPA_TEMPORAL_WEIGHT:-0.25}
repa_projection_dim=${SAM3D_REPA_PROJECTION_DIM:-768}
action_loss_weight=${ACTION_LOSS_WEIGHT:-0}
action_alignment_weight=${ACTION_ALIGNMENT_WEIGHT:-0.1}
action_feature_layer=${ACTION_FEATURE_LAYER:-7}
action_feature_layers=${ACTION_FEATURE_LAYERS:-}
action_hidden_dim=${ACTION_HIDDEN_DIM:-512}
action_architecture=${ACTION_ARCHITECTURE:-lightweight}
action_num_layers=${ACTION_NUM_LAYERS:-6}
action_num_heads=${ACTION_NUM_HEADS:-8}
action_ffn_multiplier=${ACTION_FFN_MULTIPLIER:-4}
action_pool_grid=${ACTION_POOL_GRID:-2}
action_alignment_offset=${ACTION_ALIGNMENT_OFFSET:-0}
action_validate_vector=${ACTION_VALIDATE_VECTOR:-true}
action_conditioning_enabled=${ACTION_CONDITIONING_ENABLED:-false}
action_conditioning_hidden_dim=${ACTION_CONDITIONING_HIDDEN_DIM:-8192}
action_conditioning_actions_per_latent=${ACTION_CONDITIONING_ACTIONS_PER_LATENT:-4}
action_conditioning_clip=${ACTION_CONDITIONING_CLIP:-10.0}
action_conditioning_scale=${ACTION_CONDITIONING_SCALE:-0.01}
manifest_paths=${DATASET_MANIFEST_PATHS:-}
included_batches=${INCLUDED_BATCHES:-"['legacy4k','core15k']"}
sam3d_required=${SAM3D_REQUIRED:-true}
num_frames=${NUM_FRAMES:-93}
scheduler_cycle=${SCHEDULER_CYCLE_LENGTH:-5000}
scheduler_warmup=${SCHEDULER_WARMUP_STEPS:-500}

test -d "${repo_root}"
test -d "${dataset_root}"
test -d "${cache_root}"
if [[ -n "${objects_root}" ]]; then test -d "${objects_root}"; fi
test -d "${base_checkpoint}"

manifest_overrides=()
if [[ -n "${manifest_paths}" ]]; then
  manifest_overrides=(
    "dataloader_train.dataset.manifest_paths=${manifest_paths}"
    "dataloader_train.sampler.dataset.manifest_paths=${manifest_paths}"
  )
fi

object_overrides=()
if [[ -n "${objects_root}" ]]; then
  object_overrides=(
    "dataloader_train.dataset.sam3d_objects_root=${objects_root}"
    "dataloader_train.sampler.dataset.sam3d_objects_root=${objects_root}"
  )
fi

repa_projection_overrides=()
if [[ ! "${repa_weight}" =~ ^0+([.]0+)?$ ]]; then
  repa_projection_overrides=("model.config.net.sam3d_repa_projection_dim=${repa_projection_dim}")
fi

action_overrides=()
if [[ "${action_conditioning_enabled,,}" == "true" || ! "${action_loss_weight}" =~ ^0+([.]0+)?$ ]]; then
  action_hdf5_root=${ACTION_HDF5_ROOT:-}
  action_norm_path=${ACTION_NORM_PATH:?set ACTION_NORM_PATH when action conditioning/supervision is enabled}
  if [[ -z "${action_hdf5_root}" && -z "${manifest_paths}" ]]; then
    echo "set ACTION_HDF5_ROOT or DATASET_MANIFEST_PATHS when action conditioning/supervision is enabled" >&2
    exit 2
  fi
  if [[ -n "${action_hdf5_root}" ]]; then test -d "${action_hdf5_root}"; fi
  test -s "${action_norm_path}"
  action_overrides=(
    "dataloader_train.dataset.action_required=True"
    "dataloader_train.sampler.dataset.action_required=True"
    "dataloader_train.dataset.action_norm_path=${action_norm_path}"
    "dataloader_train.sampler.dataset.action_norm_path=${action_norm_path}"
    "dataloader_train.dataset.action_alignment_offset=${action_alignment_offset}"
    "dataloader_train.sampler.dataset.action_alignment_offset=${action_alignment_offset}"
    "dataloader_train.dataset.action_validate_vector=${action_validate_vector}"
    "dataloader_train.sampler.dataset.action_validate_vector=${action_validate_vector}"
    "model.config.net.action_conditioning_enabled=${action_conditioning_enabled}"
    "model.config.net.action_conditioning_hidden_dim=${action_conditioning_hidden_dim}"
    "model.config.net.action_conditioning_actions_per_latent=${action_conditioning_actions_per_latent}"
    "model.config.net.action_conditioning_clip=${action_conditioning_clip}"
    "model.config.net.action_conditioning_scale=${action_conditioning_scale}"
  )
  if [[ -n "${action_hdf5_root}" ]]; then
    action_overrides+=(
      "dataloader_train.dataset.action_hdf5_root=${action_hdf5_root}"
      "dataloader_train.sampler.dataset.action_hdf5_root=${action_hdf5_root}"
    )
  fi
  if [[ ! "${action_loss_weight}" =~ ^0+([.]0+)?$ ]]; then
    action_overrides+=(
      "model.config.net.action_supervision_hidden_dim=${action_hidden_dim}"
      "model.config.net.action_supervision_architecture=${action_architecture}"
      "model.config.net.action_supervision_num_layers=${action_num_layers}"
      "model.config.net.action_supervision_num_heads=${action_num_heads}"
      "model.config.net.action_supervision_ffn_multiplier=${action_ffn_multiplier}"
      "model.config.net.action_supervision_pool_grid=${action_pool_grid}"
    )
    if [[ -n "${action_feature_layers}" ]]; then
      action_overrides+=("model.config.action_feature_layers=${action_feature_layers}")
    fi
  fi
fi

export COSMOS_PROJECT="${repo_root}"
export COSMOS_SAM3D_WORKSPACE="${workspace}"
export IMAGINAIRE_OUTPUT_ROOT="${workspace}/training-outputs/${run_name}"
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TORCH_MUSA_FSDP2_ENABLE_CE_COMM=${TORCH_MUSA_FSDP2_ENABLE_CE_COMM:-0}
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL=${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}

exec "${repo_root}/projects/sam3d/scripts/run_cosmos_sam3d_musa.sh" \
  -m torch.distributed.run \
  --nnodes="${nnodes}" \
  --nproc-per-node=8 \
  --node-rank="${node_rank}" \
  --master-addr="${master_addr}" \
  --master-port="${master_port}" \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_sam3d_full \
  dataloader_train.dataset.dataset_dir="${dataset_root}" \
  dataloader_train.sampler.dataset.dataset_dir="${dataset_root}" \
  "${manifest_overrides[@]}" \
  dataloader_train.dataset.sam3d_cache_dir="${cache_root}" \
  dataloader_train.sampler.dataset.sam3d_cache_dir="${cache_root}" \
  dataloader_train.dataset.sam3d_required="${sam3d_required}" \
  dataloader_train.sampler.dataset.sam3d_required="${sam3d_required}" \
  "${object_overrides[@]}" \
  dataloader_train.dataset.included_batches="${included_batches}" \
  dataloader_train.sampler.dataset.included_batches="${included_batches}" \
  dataloader_train.dataset.num_frames="${num_frames}" \
  dataloader_train.sampler.dataset.num_frames="${num_frames}" \
  dataloader_train.dataset.sampling_mode=uniform \
  dataloader_train.sampler.dataset.sampling_mode=uniform \
  dataloader_train.dataset.filter_short_videos=True \
  dataloader_train.sampler.dataset.filter_short_videos=True \
  dataloader_train.dataset.video_size='[480,640]' \
  dataloader_train.sampler.dataset.video_size='[480,640]' \
  dataloader_train.batch_size="${per_device_batch_size}" \
  dataloader_train.num_workers=2 \
  dataloader_train.persistent_workers=True \
  job.name="${run_name}" \
  job.wandb_mode="${WANDB_MODE}" \
  model.config.fsdp_shard_size=8 \
  model.config.sam3d_repa_mode="${repa_mode}" \
  model.config.sam3d_repa_weight="${repa_weight}" \
  model.config.sam3d_repa_layer="${repa_layer}" \
  model.config.sam3d_repa_temporal_weight="${repa_temporal_weight}" \
  "${repa_projection_overrides[@]}" \
  model.config.action_loss_weight="${action_loss_weight}" \
  model.config.action_alignment_weight="${action_alignment_weight}" \
  model.config.action_feature_layer="${action_feature_layer}" \
  "${action_overrides[@]}" \
  model.config.net.sam3d_teacher_tokens_as_condition=False \
  optimizer.lr="${learning_rate}" \
  optimizer.fused=False \
  scheduler.cycle_lengths="[${scheduler_cycle}]" \
  scheduler.warm_up_steps="[${scheduler_warmup}]" \
  trainer.callbacks.compile_tokenizer.enabled=False \
  trainer.callbacks.dataloader_speed.every_n=10000000 \
  trainer.callbacks.device_monitor.every_n=1 \
  trainer.callbacks.every_n_sample_reg.every_n=10000000 \
  trainer.callbacks.every_n_sample_ema.every_n=10000000 \
  trainer.max_iter="${max_iter}" \
  trainer.logging_iter=1 \
  checkpoint.load_path="${base_checkpoint}" \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  checkpoint.save_iter="${save_iter}" \
  checkpoint.dcp_async_mode_enabled=False
