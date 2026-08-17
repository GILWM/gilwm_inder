#!/usr/bin/env bash
set -euo pipefail

node_rank=${1:?usage: train_v93.sh NODE_RANK}
repo_root=${COSMOS_PROJECT:-$(git rev-parse --show-toplevel)}
workspace=${COSMOS_SAM3D_WORKSPACE:?set COSMOS_SAM3D_WORKSPACE}
dataset_root=${SAM3D_DATASET_ROOT:?set SAM3D_DATASET_ROOT}
cache_root=${SAM3D_CACHE_ROOT:?set SAM3D_CACHE_ROOT}
objects_root=${SAM3D_OBJECTS_ROOT:?set SAM3D_OBJECTS_ROOT}
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
scheduler_cycle=${SCHEDULER_CYCLE_LENGTH:-5000}
scheduler_warmup=${SCHEDULER_WARMUP_STEPS:-500}

test -d "${repo_root}"
test -d "${dataset_root}"
test -d "${cache_root}"
test -d "${objects_root}"
test -d "${base_checkpoint}"

repa_projection_overrides=()
if [[ ! "${repa_weight}" =~ ^0+([.]0+)?$ ]]; then
  repa_projection_overrides=("model.config.net.sam3d_repa_projection_dim=${repa_projection_dim}")
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
  dataloader_train.dataset.sam3d_cache_dir="${cache_root}" \
  dataloader_train.sampler.dataset.sam3d_cache_dir="${cache_root}" \
  dataloader_train.dataset.sam3d_required=True \
  dataloader_train.sampler.dataset.sam3d_required=True \
  dataloader_train.dataset.sam3d_objects_root="${objects_root}" \
  dataloader_train.sampler.dataset.sam3d_objects_root="${objects_root}" \
  dataloader_train.dataset.included_batches="['legacy4k','core15k']" \
  dataloader_train.sampler.dataset.included_batches="['legacy4k','core15k']" \
  dataloader_train.dataset.num_frames=93 \
  dataloader_train.sampler.dataset.num_frames=93 \
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
