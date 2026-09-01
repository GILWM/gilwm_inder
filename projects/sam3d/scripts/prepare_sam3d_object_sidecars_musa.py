#!/usr/bin/env python3
"""Export native SAM 3D Objects stage-1 latents and explicit geometry.

This script intentionally targets the official CUDA environment described by
facebookresearch/sam-3d-objects.  Its compact ``sam3d_objects.pt`` files are
device-independent and are merged into the MUSA Cosmos cache by
``prepare_sam3d_conditions.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mask-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam3d-repo", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="musa")
    parser.add_argument("--depth-model-path", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_video_frame(path: Path) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    frame = reader[0].asnumpy()
    del reader
    return frame


def load_cache(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary cache: {path}")
    return value


def atomic_save(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".sam3d.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize_pointmap(pointmap_H_W_3: torch.Tensor) -> torch.Tensor:
    pointmap = pointmap_H_W_3.float()
    valid = torch.isfinite(pointmap).all(dim=-1)
    clean = torch.nan_to_num(pointmap)
    if valid.any():
        center = clean[valid].median(dim=0).values
        centered = clean - center
        radius = torch.linalg.vector_norm(centered[valid], dim=-1)
        scale = torch.quantile(radius, 0.95).clamp_min(1.0e-6)
        xyz = centered / scale
    else:
        xyz = torch.zeros_like(clean)
    xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
    xyz = xyz.permute(2, 0, 1).unsqueeze(0)
    xyz = F.interpolate(xyz, (120, 160), mode="bilinear", align_corners=False)[0]
    valid_small = F.interpolate(valid[None, None].float(), (120, 160), mode="nearest")[0, 0]

    dx = F.pad(xyz[:, :, 2:] - xyz[:, :, :-2], (1, 1, 0, 0))
    dy = F.pad(xyz[:, 2:, :] - xyz[:, :-2, :], (0, 0, 1, 1))
    normals = torch.cross(dx.permute(1, 2, 0), dy.permute(1, 2, 0), dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1.0e-6).permute(2, 0, 1)
    normals = normals * valid_small.unsqueeze(0)
    depth = xyz[2:3]
    return torch.cat([depth, xyz, normals, valid_small.unsqueeze(0)], dim=0).contiguous()


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0, num-shards)")
    if not args.pipeline_config.is_file():
        raise FileNotFoundError(args.pipeline_config)
    # Instantiate the trusted, local official pipeline directly.  Importing the
    # notebook wrapper pulls in Gradio, seaborn and rendering-only packages that
    # are irrelevant to stage-1 latent extraction, and its Hydra patch attempts
    # a network download.  Direct instantiation keeps DSS preprocessing fully
    # offline while using the exact same official model and run() method.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    sys.path.insert(0, str(args.sam3d_repo))
    import sam3d_objects  # noqa: F401  # type: ignore
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    pipeline_config = OmegaConf.load(args.pipeline_config)
    pipeline_config.rendering_engine = "pytorch3d"
    pipeline_config.compile_model = False
    pipeline_config.stage1_runtime_only = True
    pipeline_config.device = args.device
    pipeline_config.depth_model.device = args.device
    if args.depth_model_path is not None:
        pipeline_config.depth_model.model.pretrained_model_name_or_path = str(args.depth_model_path)
    pipeline_config.workspace_dir = str(args.pipeline_config.parent)
    pipeline = instantiate(pipeline_config)
    records = load_records(args.manifest or args.dataset_root / "manifest.jsonl")
    end = len(records) if args.end is None else min(args.end, len(records))
    selected = [
        (index, records[index])
        for index in range(args.start, end)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    completed = skipped = failed = 0
    for position, (manifest_index, record) in enumerate(selected, 1):
        sample_dir = Path(record.get("sample_dir", f"{record['batch']}/{record['question_id']}"))
        destination = args.output_root / sample_dir / "sam3d_objects.pt"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            mask_cache_path = args.mask_cache_root / sample_dir / "condition.pt"
            if not mask_cache_path.is_file():
                skipped += 1
                print(
                    f"SAM3D_OBJECT position={position}/{len(selected)}"
                    f" id={record['question_id']} skip=missing_base_condition",
                    flush=True,
                )
                continue
            mask_cache = load_cache(mask_cache_path)
            mask_count = min(int(mask_cache["mask_valid_count"]), args.max_objects)
            image = first_video_frame(args.dataset_root / record["video"])
            masks = torch.as_tensor(mask_cache["sam_masks"], dtype=torch.float32)[:mask_count]
            if mask_count:
                masks = F.interpolate(masks[:, None], image.shape[:2], mode="nearest")[:, 0]
            print(
                "SAM3D_OBJECT_INPUT"
                f" id={record['question_id']} image={tuple(image.shape)}"
                f" masks={tuple(masks.shape)}"
                f" areas={[int(item.sum().item()) for item in masks]}",
                flush=True,
            )

            # MoGe predicts a scene point map from RGB and does not depend on
            # the per-object mask. Compute it once per video and reuse it for
            # every object instead of repeating the depth model up to 8 times.
            merged_scene = pipeline.merge_image_and_mask(image, None)
            scene_pointmap = pipeline.compute_pointmap(merged_scene)["pointmap"]
            scene_pointmap = scene_pointmap.permute(1, 2, 0).contiguous()

            shape_latents: list[torch.Tensor] = []
            object_pose: list[torch.Tensor] = []
            geometry = normalize_pointmap(scene_pointmap.detach().float().cpu())
            for object_index, mask in enumerate(masks):
                binary_mask = mask.numpy() > 0.5
                coordinates = np.argwhere(binary_mask)
                if coordinates.size == 0:
                    print(
                        f"SAM3D_OBJECT_SKIP_OBJECT id={record['question_id']} "
                        f"object={object_index} reason=empty_mask",
                        flush=True,
                    )
                    continue
                bbox_height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
                bbox_width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
                if bbox_height < 3 or bbox_width < 3:
                    print(
                        f"SAM3D_OBJECT_SKIP_OBJECT id={record['question_id']} "
                        f"object={object_index} reason=tiny_mask bbox={bbox_width}x{bbox_height}",
                        flush=True,
                    )
                    continue
                try:
                    output = pipeline.run(
                        image,
                        # The official pipeline interprets the alpha channel as
                        # uint8 image data and divides it by 255.  Passing a bool
                        # mask would therefore turn foreground into 1/255 and the
                        # point-map normalizer would see an empty object.
                        binary_mask.astype(np.uint8) * 255,
                        seed=args.seed + object_index,
                        stage1_only=True,
                        with_mesh_postprocess=False,
                        with_texture_baking=False,
                        with_layout_postprocess=False,
                        pointmap=scene_pointmap,
                    )
                except Exception as error:
                    # Degenerate masks and masks with no valid depth points are
                    # object-local data issues.  Keep the scene geometry and any
                    # other valid objects instead of dropping the whole video.
                    if isinstance(error, ZeroDivisionError) or error.__class__.__name__ == "BoundingBoxError":
                        print(
                            f"SAM3D_OBJECT_SKIP_OBJECT id={record['question_id']} "
                            f"object={object_index} reason={error.__class__.__name__}:{error}",
                            flush=True,
                        )
                        continue
                    raise
                shape = torch.as_tensor(output["shape"]).detach().float().cpu()
                if shape.ndim == 3:
                    shape = shape[0]
                if shape.ndim != 2 or shape.shape[-1] != 8:
                    raise ValueError(f"unexpected_shape_latent:{tuple(shape.shape)}")
                translation = torch.as_tensor(output["translation"]).reshape(-1, 3)[0]
                rotation = torch.as_tensor(output["rotation"]).reshape(-1, 4)[0]
                scale = torch.as_tensor(output["scale"]).reshape(-1, 3)[0]
                shape_latents.append(torch.nan_to_num(shape))
                object_pose.append(torch.nan_to_num(torch.cat([translation, rotation, scale]).float().cpu()))

            atomic_save(
                {
                    "schema_version": 1,
                    "sam3d_shape_latents": (
                        torch.stack(shape_latents)
                        if shape_latents
                        else torch.zeros(0, 256, 8, dtype=torch.float32)
                    ),
                    "sam3d_object_pose": (
                        torch.stack(object_pose)
                        if object_pose
                        else torch.zeros(0, 10, dtype=torch.float32)
                    ),
                    "sam3d_geometry": geometry,
                    "object_count": torch.tensor(len(shape_latents), dtype=torch.int64),
                    "manifest_index": manifest_index,
                    "pipeline_config": str(args.pipeline_config),
                },
                destination,
            )
            completed += 1
        except ValueError as error:
            if str(error) == "no_valid_sam3_masks":
                skipped += 1
            else:
                failed += 1
            print(f"SAM3D_OBJECT position={position}/{len(selected)} id={record['question_id']} error={error}", flush=True)
        except Exception as error:
            failed += 1
            print(f"SAM3D_OBJECT position={position}/{len(selected)} id={record['question_id']} error={error!r}", flush=True)
            traceback.print_exc()

    print(f"DONE completed={completed} skipped={skipped} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
