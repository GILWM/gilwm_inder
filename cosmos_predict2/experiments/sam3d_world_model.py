# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Experiments for the Cosmos Predict2.5 + SAM 3D branch."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.functional.lr_scheduler import LambdaWarmUpCosineScheduler
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey


DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]
DATASET_ROOT = "/datahdd/mccxadmin/train_data"
SAM3D_CACHE_ROOT = "/datahdd/mccxadmin/cosmos-sam3d-cache/v1"

sam3d_dataset = L(SAM3DVideoDataset)(
    dataset_dir=DATASET_ROOT,
    sam3d_cache_dir=SAM3D_CACHE_ROOT,
    sam3d_required=True,
    # Formal training must use the atomically generated, deeply validated
    # allowlist. Teacher/DINO tokens are deliberately not part of eligibility.
    sam3d_require_complete_conditions=True,
    sam3d_load_teacher_tokens=False,
    included_batches=["legacy4k", "core15k"],
    num_frames=121,
    video_size=(480, 640),
    sampling_mode="uniform",
    filter_short_videos=True,
    sam3d_teacher_frames=8,
    sam3d_num_tokens=256,
    sam3d_token_dim=768,
    sam_mask_instances=8,
    sam_mask_size=(120, 160),
    sam_mask_meta_dim=8,
    sam3d_geometry_channels=8,
    sam3d_geometry_size=(120, 160),
    sam3d_shape_instances=8,
    sam3d_shape_tokens=256,
    sam3d_shape_dim=8,
    sam3d_pose_dim=10,
    repeat_factor=1,
    sam3d_native_index=None,
    sam3d_objects_root=None,
)

sam3d_dataloader = L(get_generic_dataloader)(
    dataset=sam3d_dataset,
    sampler=L(get_sampler)(dataset=sam3d_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

sam3d_cosine_scheduler = L(LambdaWarmUpCosineScheduler)(
    warm_up_steps=[3_000],
    cycle_lengths=[30_000],
    f_start=[1.0e-6],
    f_max=[1.0],
    f_min=[0.1],
)


def _base_experiment(name: str, max_iter: int, dataset, dataloader) -> dict:
    return dict(
        defaults=[
            f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
            {"override /model": "fsdp_sam3d_rectified_flow"},
            {"override /net": "cosmos_v1_2B_sam3d"},
            {"override /conditioner": "sam3d_video_prediction_conditioner"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict_v2p5_sam3d",
            group="video2world",
            name=name,
        ),
        dataloader_train=dataloader,
        checkpoint=dict(
            save_iter=500,
            load_path=DEFAULT_CHECKPOINT.s3.uri,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        optimizer=dict(
            lr=3.0e-5,
            weight_decay=0.001,
            fused=False,
        ),
        scheduler=sam3d_cosine_scheduler,
        trainer=dict(
            logging_iter=1,
            max_iter=max_iter,
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=200, save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(every_n=500, save_s3=False),
                every_n_sample_ema=dict(every_n=500, save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
    )


predict2_video2world_training_2b_sam3d_full = _base_experiment(
    "2b_sam3d_full_30k",
    30_000,
    sam3d_dataset,
    sam3d_dataloader,
)

smoke_dataset = copy.deepcopy(sam3d_dataset)
smoke_dataset.sam3d_required = False
smoke_dataset.sam3d_require_complete_conditions = False
smoke_dataset.included_batches = ["legacy4k"]
smoke_dataloader = copy.deepcopy(sam3d_dataloader)
smoke_dataloader.dataset = smoke_dataset
smoke_dataloader.sampler.dataset = smoke_dataset
smoke_dataloader.batch_size = 1
smoke_dataloader.num_workers = 1
smoke_dataloader.persistent_workers = False

predict2_video2world_training_2b_sam3d_smoke = _base_experiment(
    "2b_sam3d_smoke",
    10,
    smoke_dataset,
    smoke_dataloader,
)
predict2_video2world_training_2b_sam3d_smoke["checkpoint"]["save_iter"] = 10


cs = ConfigStore.instance()
for _item in (
    predict2_video2world_training_2b_sam3d_full,
    predict2_video2world_training_2b_sam3d_smoke,
):
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(group="experiment", package="_global_", name=experiment_name, node=_item)
