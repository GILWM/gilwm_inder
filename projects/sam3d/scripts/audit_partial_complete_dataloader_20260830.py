#!/usr/bin/env python3
"""Audit the frozen complete-condition subset before distributed training."""

from pathlib import Path

import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset


DATA = "/datahdd/mccxadmin/train_data/sam3d_mix_v93_wa2_robotwin_core_flow_20260827"
CACHE = "/datahdd/mccxadmin/train_data/.sam3d_cache/sam3d_mix_v93_wa2_robotwin_core_flow_20260827"
OBJECTS = "/datahdd/mccxadmin/cosmos-sam3d-sidecars/sam3d_mix_v93_wa2_robotwin_core_flow_20260827"
INDEX = f"{OBJECTS}/native_complete_frozen_20260830.txt"
BATCHES = [
    "worldarena2_wa2_50k_remaining",
    "robotwin2_14d",
    "core15k",
    "flowwam_wa2_seed2_20260823",
    "flowwam_wa2_seed3_20260823",
    "flowwam_wa2_stage2_20260810",
]


def build_dataset() -> SAM3DVideoDataset:
    return SAM3DVideoDataset(
        dataset_dir=DATA,
        sam3d_cache_dir=CACHE,
        sam3d_objects_root=OBJECTS,
        sam3d_native_index=INDEX,
        sam3d_required=True,
        sam3d_require_complete_conditions=True,
        sam3d_load_teacher_tokens=False,
        included_batches=BATCHES,
        num_frames=93,
        video_size=(480, 640),
        sampling_mode="uniform",
        filter_short_videos=True,
        repeat_factor=1,
    )


def main() -> None:
    index_entries = [line for line in Path(INDEX).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(index_entries) != len(set(index_entries)):
        raise ValueError("Frozen native index contains duplicate samples")
    dataset = build_dataset()
    if not dataset.video_paths:
        raise ValueError("Complete-condition filtering produced an empty dataset")

    probe_count = min(64, len(dataset))
    probe_indices = torch.linspace(0, len(dataset) - 1, probe_count).round().long().tolist()
    expected_shapes = {
        "sam3d_tokens_B_F_N_D": (1, 1, 768),
        "sam_mask_B_K_H_W": (8, 120, 160),
        "sam_mask_meta_B_K_D": (8, 8),
        "sam3d_geometry_B_C_H_W": (8, 120, 160),
        "sam3d_shape_latents_B_K_N_D": (8, 256, 8),
        "sam3d_object_pose_B_K_D": (8, 10),
    }
    probed = []
    for index in probe_indices:
        condition = dataset._load_condition(index)
        for key, shape in expected_shapes.items():
            tensor = condition[key]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"sample={index} {key} shape={tuple(tensor.shape)} expected={shape}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"sample={index} {key} contains NaN/Inf")
        if torch.count_nonzero(condition["sam3d_tokens_B_F_N_D"]):
            raise ValueError(f"sample={index} unexpectedly contains teacher tokens")
        if not torch.count_nonzero(condition["sam3d_geometry_B_C_H_W"]):
            raise ValueError(f"sample={index} has zero SAM3D geometry")
        probed.append(condition)

    # Exercise the default collation shape for the exact per-device batch size.
    for key in expected_shapes:
        batch = torch.stack([value[key] for value in probed[:4]])
        if batch.shape[0] != 4:
            raise ValueError(f"Failed to collate {key}")

    print(
        "PARTIAL_COMPLETE_DATALOADER_OK",
        f"validated_index_entries={len(index_entries)}",
        f"eligible_after_video_and_condition_filters={len(dataset)}",
        f"probed={len(probed)}",
        "batch_size=4",
        "frames=93",
        "sampling=uniform",
        "teacher_tokens=false",
    )


if __name__ == "__main__":
    main()
