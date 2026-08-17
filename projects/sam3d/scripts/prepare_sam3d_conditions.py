#!/usr/bin/env python3
"""Build fixed-shape SAM 3 + SAM 3D condition caches for Cosmos training.

The SAM 3D token branch is the frozen DINOv2 condition encoder used by the
official SAM 3D Objects pipeline.  Full rendered geometry (depth/point map/
normals/confidence) can be merged from a sidecar produced by the gated SAM 3D
Objects checkpoint without changing the training dataset format.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image


MASK_SIZE = (120, 160)
GEOMETRY_CHANNELS = 8  # depth, xyz(3), normals(3), confidence
GEOMETRY_SIZE = (120, 160)


def resize_nearest_2d(value: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize ``[..., H, W]`` without the incompatible MUSA nearest wrapper."""

    target_h, target_w = (int(target_size[0]), int(target_size[1]))
    input_h, input_w = value.shape[-2:]
    if (input_h, input_w) == (target_h, target_w):
        return value
    y_indices = torch.linspace(0, input_h - 1, target_h).round().long()
    x_indices = torch.linspace(0, input_w - 1, target_w).round().long()
    return value.index_select(-2, y_indices).index_select(-1, x_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/datahdd/mccxadmin/train_data"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("/datahdd/mccxadmin/cosmos-sam3d-cache/v1"))
    parser.add_argument("--sam3-checkpoint", type=Path, default=Path(os.environ.get("SAM3_CHECKPOINT", "")))
    parser.add_argument(
        "--dino-repo",
        type=Path,
        default=Path("/datassd/morka/cosmos-sam3d-work/third_party/dinov2-official"),
    )
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument(
        "--sam3d-objects-root",
        type=Path,
        help="Optional root containing native SAM 3D Objects stage-1 sidecars",
    )
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--teacher-frames", type=int, default=8)
    parser.add_argument("--mask-instances", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.25)
    parser.add_argument("--fallback-prompts", default="robot,object")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-frames", type=int, default=121)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dino-only", action="store_true", help="Generate condition tokens without SAM 3 masks")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("batch", "question_id", "video", "instruction"):
                if key not in record:
                    raise ValueError(f"manifest line {line_number} lacks {key}: {record}")
            records.append(record)
    return records


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> Iterable[tuple[int, dict[str, Any]]]:
    end = len(records) if args.end is None else min(args.end, len(records))
    selected = [(index, records[index]) for index in range(args.start, end)]
    selected = [item for item in selected if item[0] % args.num_shards == args.shard_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def read_instruction(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("instruction"), str):
            return value["instruction"].strip()
        for nested in value.values():
            if isinstance(nested, str):
                return nested.strip()
            if isinstance(nested, dict):
                for candidate in ("short", "medium", "long", "instruction"):
                    if isinstance(nested.get(candidate), str):
                        return nested[candidate].strip()
    return ""


def load_video_frames(path: Path, count: int, minimum_frames: int) -> tuple[list[Image.Image], list[int], int]:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    total = len(reader)
    if total < minimum_frames:
        raise ValueError(f"short_video:{total}<{minimum_frames}")
    indices = np.rint(np.linspace(0, total - 1, count)).astype(np.int64).tolist()
    indices[0] = 0
    arrays = reader.get_batch(indices).asnumpy()
    del reader
    return [Image.fromarray(array) for array in arrays], indices, total


class Sam3DConditionEncoder:
    """Official SAM 3D Objects DINOv2 image-condition encoder."""

    def __init__(self, repo: Path, device: str):
        self.device = torch.device(device)
        self.model = torch.hub.load(
            str(repo), "dinov2_vitb14", source="local", pretrained=True, verbose=False
        ).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        batch = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images])
        value = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
        value = F.interpolate(value, (224, 224), mode="bilinear", align_corners=False)
        value = (value - self.mean) / self.std
        output = self.model.forward_features(value)
        # SAM 3D's Dino._forward_last_layer emits CLS + patch tokens.  The
        # 16x16 patch grid is the spatial teacher signal used by our REPA loss.
        tokens = output["x_norm_patchtokens"]
        if tokens.shape[1:] != (256, 768):
            raise ValueError(f"Unexpected DINOv2 token shape: {tuple(tokens.shape)}")
        return tokens.float().cpu().contiguous()


class Sam3MaskEncoder:
    def __init__(self, checkpoint: Path, device: str, threshold: float):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device
        self.model = build_sam3_image_model(
            device=device,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=threshold)

    @staticmethod
    def _iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        intersection = torch.logical_and(mask_a, mask_b).sum().item()
        union = torch.logical_or(mask_a, mask_b).sum().item()
        return intersection / max(union, 1)

    @torch.inference_mode()
    def __call__(self, image: Image.Image, prompts: list[str], limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.processor.set_image(image)
        candidates: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
        for prompt_index, prompt in enumerate(prompts):
            if not prompt:
                continue
            output = self.processor.set_text_prompt(prompt=prompt[:512], state=state)
            masks = output["masks"].detach().bool().cpu()
            boxes = output["boxes"].detach().float().cpu()
            scores = output["scores"].detach().float().cpu()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            for mask, box, score in zip(masks, boxes, scores):
                candidates.append((float(score), prompt_index, mask, box))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
        for candidate in candidates:
            if any(self._iou(candidate[2], previous[2]) > 0.90 for previous in selected):
                continue
            selected.append(candidate)
            if len(selected) == limit:
                break

        height, width = image.height, image.width
        masks_out = torch.zeros(limit, *MASK_SIZE, dtype=torch.float32)
        meta_out = torch.zeros(limit, 8, dtype=torch.float32)
        for index, (score, prompt_index, mask, box) in enumerate(selected):
            resized = resize_nearest_2d(mask.float(), MASK_SIZE)
            masks_out[index] = resized
            x0, y0, x1, y1 = box.tolist()
            meta_out[index] = torch.tensor(
                [
                    x0 / width,
                    y0 / height,
                    x1 / width,
                    y1 / height,
                    score,
                    float(mask.float().mean()),
                    float(prompt_index),
                    1.0,
                ]
            )
        return masks_out, meta_out


def geometry_sidecar(root: Path | None, record: dict[str, Any]) -> tuple[torch.Tensor, bool, str | None]:
    empty = torch.zeros(GEOMETRY_CHANNELS, *GEOMETRY_SIZE, dtype=torch.float32)
    if root is None:
        return empty, False, None
    sample_dir = Path(record.get("sample_dir", f"{record['batch']}/{record['question_id']}"))
    candidates = (
        root / sample_dir / "geometry.pt",
        root / sample_dir / "sam3d_objects.pt",
        root / record["batch"] / f"{record['question_id']}.pt",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return empty, False, None
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("sam3d_geometry", value.get("geometry"))
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != GEOMETRY_CHANNELS:
        raise ValueError(f"Geometry sidecar {path} must be [8,H,W], got {tuple(value.shape)}")
    value = F.interpolate(value.unsqueeze(0), GEOMETRY_SIZE, mode="bilinear", align_corners=False)[0]
    return value.contiguous(), True, str(path)


def sam3d_objects_sidecar(
    root: Path | None, record: dict[str, Any], instances: int, tokens: int
) -> tuple[torch.Tensor, torch.Tensor, int, str | None]:
    empty_latents = torch.zeros(instances, tokens, 8, dtype=torch.float32)
    empty_pose = torch.zeros(instances, 10, dtype=torch.float32)
    if root is None:
        return empty_latents, empty_pose, 0, None
    sample_dir = Path(record.get("sample_dir", f"{record['batch']}/{record['question_id']}"))
    candidates = (
        root / sample_dir / "sam3d_objects.pt",
        root / sample_dir / "condition_3d.pt",
        root / record["batch"] / f"{record['question_id']}.pt",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return empty_latents, empty_pose, 0, None
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"SAM 3D Objects sidecar {path} must be a dictionary")
    shape_latents = torch.as_tensor(value["sam3d_shape_latents"], dtype=torch.float32)
    object_pose = torch.as_tensor(value["sam3d_object_pose"], dtype=torch.float32)
    if shape_latents.ndim == 2:
        shape_latents = shape_latents.unsqueeze(0)
    if shape_latents.ndim != 3 or shape_latents.shape[-1] != 8:
        raise ValueError(f"{path}: sam3d_shape_latents must be [K,N,8], got {tuple(shape_latents.shape)}")
    if object_pose.ndim == 1:
        object_pose = object_pose.unsqueeze(0)
    if object_pose.ndim != 2 or object_pose.shape[-1] != 10:
        raise ValueError(f"{path}: sam3d_object_pose must be [K,10], got {tuple(object_pose.shape)}")
    valid = min(instances, shape_latents.shape[0], object_pose.shape[0])
    if valid:
        resampled = F.adaptive_avg_pool1d(shape_latents[:valid].transpose(1, 2), tokens).transpose(1, 2)
        empty_latents[:valid] = resampled
        empty_pose[:valid] = object_pose[:valid]
    return empty_latents.contiguous(), empty_pose.contiguous(), valid, str(path)


def atomic_torch_save(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".condition.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0, num-shards)")
    manifest = args.manifest or args.dataset_root / "manifest.jsonl"
    if not args.dino_repo.joinpath("hubconf.py").is_file():
        raise FileNotFoundError(args.dino_repo / "hubconf.py")
    if not args.dino_only and not args.sam3_checkpoint.is_file():
        raise FileNotFoundError(args.sam3_checkpoint)

    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401

    dino = Sam3DConditionEncoder(args.dino_repo, args.device)
    sam3 = None if args.dino_only else Sam3MaskEncoder(args.sam3_checkpoint, args.device, args.mask_threshold)
    fallback_prompts = [prompt.strip() for prompt in args.fallback_prompts.split(",") if prompt.strip()]
    records = load_records(manifest)
    selected = list(select_records(records, args))
    completed = skipped = failed = 0
    started = time.time()

    for position, (manifest_index, record) in enumerate(selected, 1):
        sample_dir = Path(record.get("sample_dir", f"{record['batch']}/{record['question_id']}"))
        destination = args.cache_root / sample_dir / "condition.pt"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            frames, frame_indices, source_frame_count = load_video_frames(
                args.dataset_root / record["video"], args.teacher_frames, args.minimum_frames
            )
            tokens = dino(frames)
            instruction = read_instruction(args.dataset_root / record["instruction"])
            prompts = [instruction, *fallback_prompts]
            if sam3 is None:
                masks = torch.zeros(args.mask_instances, *MASK_SIZE, dtype=torch.float32)
                mask_meta = torch.zeros(args.mask_instances, 8, dtype=torch.float32)
            else:
                masks, mask_meta = sam3(frames[0], prompts, args.mask_instances)
            geometry, geometry_valid, geometry_source = geometry_sidecar(
                args.geometry_root or args.sam3d_objects_root, record
            )
            shape_latents, object_pose, object_valid_count, object_source = sam3d_objects_sidecar(
                args.sam3d_objects_root, record, args.mask_instances, 256
            )
            cache = {
                "schema_version": 2,
                "sam3d_tokens": tokens,
                "sam_masks": masks,
                "sam_mask_meta": mask_meta,
                "sam3d_geometry": geometry,
                "sam3d_shape_latents": shape_latents,
                "sam3d_object_pose": object_pose,
                "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
                "source_frame_count": torch.tensor(source_frame_count, dtype=torch.int64),
                "mask_valid_count": torch.tensor(int(mask_meta[:, 7].sum().item()), dtype=torch.int64),
                "geometry_valid": torch.tensor(geometry_valid, dtype=torch.bool),
                "sam3d_object_valid_count": torch.tensor(object_valid_count, dtype=torch.int64),
                "provenance": {
                    "sam3": "facebook/sam3" if sam3 is not None else "disabled",
                    "sam3d_token_encoder": "facebookresearch/dinov2:dinov2_vitb14 (SAM 3D Objects conditioner)",
                    "geometry_source": geometry_source or "missing",
                    "sam3d_objects_source": object_source or "missing",
                },
                "manifest_index": manifest_index,
            }
            atomic_torch_save(cache, destination)
            completed += 1
        except ValueError as error:
            if str(error).startswith("short_video:"):
                skipped += 1
                print(f"SKIP index={manifest_index} id={record['question_id']} reason={error}", flush=True)
                continue
            failed += 1
            print(f"FAIL index={manifest_index} id={record['question_id']} error={error!r}", flush=True)
        except Exception as error:
            failed += 1
            print(f"FAIL index={manifest_index} id={record['question_id']} error={error!r}", flush=True)

        elapsed = max(time.time() - started, 1e-6)
        print(
            f"PROGRESS shard={args.shard_index}/{args.num_shards} position={position}/{len(selected)} "
            f"completed={completed} skipped={skipped} failed={failed} samples_per_min={completed / elapsed * 60:.2f}",
            flush=True,
        )

    print(
        f"DONE shard={args.shard_index}/{args.num_shards} completed={completed} skipped={skipped} failed={failed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
