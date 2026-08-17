#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
project="${workspace}/cosmos-predict2.5-sam3d-musa"
cache_root="${SAM3D_CACHE_ROOT:-/datahdd/mccxadmin/cosmos-sam3d-cache/real-smoke}"
run_name="${SAM3D_RUN_NAME:-smoke-real-sam3-multitoken}"
max_iter="${SAM3D_MAX_ITER:-1}"

export IMAGINAIRE_OUTPUT_ROOT="${workspace}/training-outputs/${run_name}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export TORCH_MUSA_FSDP2_ENABLE_CE_COMM=0
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL=0

cd "${project}"
exec "${workspace}/scripts/run_cosmos_sam3d_musa.sh" \
  -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_sam3d_smoke \
  dataloader_train.dataset.sam3d_cache_dir="${cache_root}" \
  dataloader_train.sampler.dataset.sam3d_cache_dir="${cache_root}" \
  dataloader_train.dataset.sam3d_required=True \
  dataloader_train.sampler.dataset.sam3d_required=True \
  dataloader_train.num_workers=1 \
  dataloader_train.persistent_workers=False \
  job.name="${run_name}" \
  job.wandb_mode=disabled \
  model.config.fsdp_shard_size=8 \
  optimizer.fused=False \
  trainer.max_iter="${max_iter}" \
  trainer.logging_iter=1 \
  trainer.callbacks.dataloader_speed.every_n=10000000 \
  trainer.callbacks.device_monitor.every_n=10000000 \
  trainer.callbacks.every_n_sample_reg.every_n=10000000 \
  trainer.callbacks.every_n_sample_ema.every_n=10000000 \
  checkpoint.save_iter="${max_iter}" \
  checkpoint.dcp_async_mode_enabled=False
