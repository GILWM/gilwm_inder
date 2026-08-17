#!/usr/bin/env bash
set -euo pipefail

node_rank=${1:?usage: run_sam3d_v93_train.sh NODE_RANK smoke|train5000}
mode=${2:-smoke}
workspace=/datassd/morka/cosmos-sam3d-work
project=${workspace}/cosmos-predict2.5-sam3d-musa
data=/datahdd/mccxadmin/train_data
master_addr=${MASTER_ADDR:-172.11.0.25}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}
repa_mode=${SAM3D_REPA_MODE:-projected_cosine}
repa_weight=${SAM3D_REPA_WEIGHT:-0}
repa_layer=${SAM3D_REPA_LAYER:-7}
repa_temporal_weight=${SAM3D_REPA_TEMPORAL_WEIGHT:-0.25}
repa_projection_dim=${SAM3D_REPA_PROJECTION_DIM:-768}
teacher_tokens_as_condition=${SAM3D_TEACHER_TOKENS_AS_CONDITION:-false}
scheduler_cycle=${SCHEDULER_CYCLE_LENGTH:-5000}
scheduler_warmup=${SCHEDULER_WARMUP_STEPS:-500}
init_checkpoint=${SAM3D_INIT_CHECKPOINT:-}

case "$mode" in
  smoke)
    nnodes=1
    max_iter=${MAX_ITER:-3}
    save_iter=$max_iter
    repeat_factor=${REPEAT_FACTOR:-32}
    cache_root=${SAM3D_CACHE_ROOT:-/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93-small}
    native_index=${SAM3D_NATIVE_INDEX:-/datahdd/mccxadmin/cosmos-sam3d-native-index/core15k-legacy4k-v93-small.txt}
    objects_root=${SAM3D_OBJECTS_ROOT:-/datahdd/mccxadmin/cosmos-sam3d-sidecars/core15k-legacy4k-v93-small}
    master_port=${MASTER_PORT:-29931}
    run_name=${SAM3D_RUN_NAME:-sam3d-v93-small-bs${per_device_batch_size}}
    ;;
  train5000)
    nnodes=4
    max_iter=${MAX_ITER:-5000}
    save_iter=${SAVE_ITER:-500}
    repeat_factor=1
    cache_root=${SAM3D_CACHE_ROOT:-/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93}
    # Keep every >=93-frame Core15K/Legacy sample in the formal run. Samples
    # without a valid object mask retain DINO conditioning and zero native
    # shape slots instead of being silently removed from the dataset.
    native_index=${SAM3D_NATIVE_INDEX:-none}
    objects_root=${SAM3D_OBJECTS_ROOT:-/datahdd/mccxadmin/cosmos-sam3d-sidecars/core15k-legacy4k-v93}
    master_port=${MASTER_PORT:-29932}
    run_name=${SAM3D_RUN_NAME:-sam3d-v93-core15k-legacy4k-bs${per_device_batch_size}-5k}
    ;;
  *)
    echo "mode must be smoke or train5000" >&2
    exit 2
    ;;
esac

native_index_overrides=()
if [[ -n "$native_index" && "$native_index" != "none" ]]; then
  test -s "$native_index"
  native_index_overrides=(
    "dataloader_train.dataset.sam3d_native_index=$native_index"
    "dataloader_train.sampler.dataset.sam3d_native_index=$native_index"
  )
fi
repa_projection_overrides=()
# Do not instantiate/checkpoint an unused projection head in the default
# no-REPA configuration.  Explicit REPA experiments opt in by setting a
# positive SAM3D_REPA_WEIGHT.
if [[ ! "$repa_weight" =~ ^0+([.]0+)?$ ]]; then
  repa_projection_overrides=("model.config.net.sam3d_repa_projection_dim=$repa_projection_dim")
fi
init_checkpoint_overrides=()
if [[ -n "$init_checkpoint" ]]; then
  if [[ "$init_checkpoint" == /* ]]; then
    test -d "$init_checkpoint"
  fi
  init_checkpoint_overrides=(
    "checkpoint.load_path=$init_checkpoint"
    "checkpoint.load_training_state=False"
    "checkpoint.strict_resume=False"
  )
fi
export IMAGINAIRE_OUTPUT_ROOT=${workspace}/training-outputs/${run_name}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export TORCH_MUSA_FSDP2_ENABLE_CE_COMM=0
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL=0

cd "$project"
exec "${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  -m torch.distributed.run \
  --nnodes="$nnodes" \
  --nproc-per-node=8 \
  --node-rank="$node_rank" \
  --master-addr="$master_addr" \
  --master-port="$master_port" \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_sam3d_full \
  dataloader_train.dataset.dataset_dir="$data" \
  dataloader_train.sampler.dataset.dataset_dir="$data" \
  dataloader_train.dataset.sam3d_cache_dir="$cache_root" \
  dataloader_train.sampler.dataset.sam3d_cache_dir="$cache_root" \
  dataloader_train.dataset.sam3d_required=True \
  dataloader_train.sampler.dataset.sam3d_required=True \
  dataloader_train.dataset.sam3d_objects_root="$objects_root" \
  dataloader_train.sampler.dataset.sam3d_objects_root="$objects_root" \
  "${native_index_overrides[@]}" \
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
  dataloader_train.dataset.repeat_factor="$repeat_factor" \
  dataloader_train.sampler.dataset.repeat_factor="$repeat_factor" \
  dataloader_train.batch_size="$per_device_batch_size" \
  dataloader_train.num_workers=2 \
  dataloader_train.persistent_workers=True \
  job.name="$run_name" \
  job.wandb_mode=disabled \
  model.config.fsdp_shard_size=8 \
  model.config.sam3d_repa_mode="$repa_mode" \
  model.config.sam3d_repa_weight="$repa_weight" \
  model.config.sam3d_repa_layer="$repa_layer" \
  model.config.sam3d_repa_temporal_weight="$repa_temporal_weight" \
  "${repa_projection_overrides[@]}" \
  model.config.net.sam3d_teacher_tokens_as_condition="$teacher_tokens_as_condition" \
  optimizer.fused=False \
  scheduler.cycle_lengths="[$scheduler_cycle]" \
  scheduler.warm_up_steps="[$scheduler_warmup]" \
  trainer.callbacks.compile_tokenizer.enabled=False \
  trainer.callbacks.dataloader_speed.every_n=10000000 \
  trainer.callbacks.device_monitor.every_n=1 \
  trainer.callbacks.every_n_sample_reg.every_n=10000000 \
  trainer.callbacks.every_n_sample_ema.every_n=10000000 \
  trainer.max_iter="$max_iter" \
  trainer.logging_iter=1 \
  checkpoint.save_iter="$save_iter" \
  "${init_checkpoint_overrides[@]}" \
  checkpoint.dcp_async_mode_enabled=False
