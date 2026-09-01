# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Video dataset that attaches SAM 3 masks and native SAM 3D conditions."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset


_LOGGED_ELIGIBILITY_SUMMARIES: set[tuple[str, str, str]] = set()


class SAM3DVideoDataset(VideoDataset):
    """Load ordinary Cosmos samples plus offline SAM 3 / SAM 3D conditions.

    Cache layout mirrors the compact training-data layout::

        CACHE_ROOT/<batch>/<question_id>/condition.pt

    Teacher tokens are optional and disabled by default. Shapes are normalized
    here so the default DataLoader collate function remains deterministic.
    """

    def __init__(
        self,
        *args,
        sam3d_cache_dir: str,
        sam3d_required: bool = True,
        sam3d_load_teacher_tokens: bool = False,
        sam3d_teacher_frames: int = 8,
        sam3d_num_tokens: int = 256,
        sam3d_token_dim: int = 768,
        sam_mask_instances: int = 8,
        sam_mask_size: tuple[int, int] = (120, 160),
        sam_mask_meta_dim: int = 8,
        sam3d_geometry_channels: int = 8,
        sam3d_geometry_size: tuple[int, int] = (120, 160),
        sam3d_shape_instances: int = 8,
        sam3d_shape_tokens: int = 256,
        sam3d_shape_dim: int = 8,
        sam3d_pose_dim: int = 10,
        repeat_factor: int = 1,
        sam3d_native_index: Optional[str] = None,
        sam3d_objects_root: Optional[str] = None,
        sam3d_require_complete_conditions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sam3d_cache_dir = sam3d_cache_dir
        self.sam3d_required = sam3d_required
        self.sam3d_load_teacher_tokens = sam3d_load_teacher_tokens
        self.sam3d_teacher_frames = sam3d_teacher_frames
        self.sam3d_num_tokens = sam3d_num_tokens
        self.sam3d_token_dim = sam3d_token_dim
        self.sam_mask_instances = sam_mask_instances
        self.sam_mask_size = sam_mask_size
        self.sam_mask_meta_dim = sam_mask_meta_dim
        self.sam3d_geometry_channels = sam3d_geometry_channels
        self.sam3d_geometry_size = sam3d_geometry_size
        self.sam3d_shape_instances = sam3d_shape_instances
        self.sam3d_shape_tokens = sam3d_shape_tokens
        self.sam3d_shape_dim = sam3d_shape_dim
        self.sam3d_pose_dim = sam3d_pose_dim
        self.sam3d_objects_root = sam3d_objects_root
        self.sam3d_require_complete_conditions = sam3d_require_complete_conditions
        if repeat_factor < 1:
            raise ValueError(f"repeat_factor must be >= 1, got {repeat_factor}")
        if self.sam3d_require_complete_conditions:
            if not self.sam3d_required:
                raise ValueError("sam3d_require_complete_conditions=True requires sam3d_required=True")
            if sam3d_native_index is None:
                raise ValueError(
                    "sam3d_require_complete_conditions=True requires a validated sam3d_native_index"
                )
            if self.sam3d_objects_root is None:
                raise ValueError(
                    "sam3d_require_complete_conditions=True requires sam3d_objects_root"
                )

        candidate_count = len(self.video_paths)
        missing_cache_count = 0
        self.sam3d_condition_paths = [self._condition_path(video_path) for video_path in self.video_paths]
        if self.sam3d_required:
            keep_indices = [index for index, path in enumerate(self.sam3d_condition_paths) if os.path.isfile(path)]
            missing_cache_count = len(self.video_paths) - len(keep_indices)
            self._keep_samples(keep_indices)
            log.info(
                f"SAM 3D cache coverage: kept {len(keep_indices)} samples and "
                f"excluded {missing_cache_count} missing caches"
            )
            if not self.video_paths:
                raise ValueError(f"No SAM 3D caches found under {self.sam3d_cache_dir}")

        not_in_native_index_count = 0
        native_sample_count = 0
        if sam3d_native_index is not None:
            index_path = Path(sam3d_native_index)
            if not index_path.is_file():
                raise FileNotFoundError(index_path)
            native_samples = {
                line.strip().replace("\\", "/")
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            native_sample_count = len(native_samples)
            keep_indices = []
            for index, condition_path in enumerate(self.sam3d_condition_paths):
                sample = os.path.relpath(os.path.dirname(condition_path), self.sam3d_cache_dir).replace("\\", "/")
                if sample in native_samples:
                    keep_indices.append(index)
            not_in_native_index_count = len(self.video_paths) - len(keep_indices)
            self._keep_samples(keep_indices)
            log.info(
                f"Native SAM 3D index: kept {len(keep_indices)} samples from "
                f"{len(native_samples)} indexed sidecars"
            )
            if not self.video_paths:
                raise ValueError(f"No dataset samples matched native SAM 3D index {index_path}")

        self.sam3d_object_paths = [self._object_path(path) for path in self.sam3d_condition_paths]
        missing_sidecar_count = 0
        if self.sam3d_require_complete_conditions:
            keep_indices = [
                index
                for index, path in enumerate(self.sam3d_object_paths)
                if path is not None and os.path.isfile(path)
            ]
            missing_sidecar_count = len(self.video_paths) - len(keep_indices)
            self._keep_samples(keep_indices, include_object_paths=True)
            if not self.video_paths:
                raise ValueError(
                    "No samples have complete SAM 3 / SAM 3D conditions after applying "
                    f"{sam3d_native_index}"
                )

        if self.sam3d_objects_root is not None:
            native_count = sum(os.path.isfile(path) for path in self.sam3d_object_paths)
            log.info(
                f"Native SAM 3D sidecar overlay: found {native_count} of "
                f"{len(self.sam3d_object_paths)} dataset samples under {self.sam3d_objects_root}"
            )

        if self.sam3d_require_complete_conditions:
            summary_key = (
                os.path.realpath(self.dataset_dir),
                os.path.realpath(self.sam3d_cache_dir),
                os.path.realpath(str(sam3d_native_index)),
            )
            if summary_key not in _LOGGED_ELIGIBILITY_SUMMARIES:
                _LOGGED_ELIGIBILITY_SUMMARIES.add(summary_key)
                filtered_count = candidate_count - len(self.video_paths)
                log.info(
                    "SAM3D_TRAINING_ELIGIBILITY "
                    f"candidates_after_video_filters={candidate_count} "
                    f"eligible_complete={len(self.video_paths)} filtered_total={filtered_count} "
                    f"missing_base_cache={missing_cache_count} "
                    f"not_in_validated_index={not_in_native_index_count} "
                    f"missing_sidecar={missing_sidecar_count} "
                    f"validated_index_entries={native_sample_count} teacher_tokens_required=false",
                    rank0_only=True,
                )

        # Small-scale memory/stability tests deliberately use a handful of
        # fully native sidecars.  Logical repetition supplies enough local
        # samples for large per-device batches without copying cache files or
        # changing full-scale sampling, where repeat_factor remains one.
        if repeat_factor > 1:
            self.video_paths = self.video_paths * repeat_factor
            self.sam3d_condition_paths = self.sam3d_condition_paths * repeat_factor
            self.sam3d_object_paths = self.sam3d_object_paths * repeat_factor
            if self.caption_paths is not None:
                self.caption_paths = self.caption_paths * repeat_factor
            log.info(f"Repeated SAM 3D dataset {repeat_factor}x for {len(self.video_paths)} logical samples")

    def _keep_samples(self, keep_indices: list[int], include_object_paths: bool = False) -> None:
        """Apply one eligibility filter to all parallel sample-path lists."""

        self.video_paths = [self.video_paths[index] for index in keep_indices]
        self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
        if self.caption_paths is not None:
            self.caption_paths = [self.caption_paths[index] for index in keep_indices]
        if include_object_paths:
            self.sam3d_object_paths = [self.sam3d_object_paths[index] for index in keep_indices]

    def _condition_path(self, video_path: str) -> str:
        relative_video_path = os.path.relpath(video_path, self.dataset_dir)
        return os.path.join(self.sam3d_cache_dir, os.path.dirname(relative_video_path), "condition.pt")

    def _object_path(self, condition_path: str) -> Optional[str]:
        if self.sam3d_objects_root is None:
            return None
        sample = os.path.relpath(os.path.dirname(condition_path), self.sam3d_cache_dir)
        return os.path.join(self.sam3d_objects_root, sample, "sam3d_objects.pt")

    @staticmethod
    def _load_tensor_cache(path: str) -> dict[str, Any]:
        try:
            cache = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            cache = torch.load(path, map_location="cpu")
        if not isinstance(cache, dict):
            raise TypeError(f"Expected a tensor dictionary in {path}, got {type(cache)}")
        return cache

    @staticmethod
    def _select_or_pad_axis(value: torch.Tensor, target: int, axis: int) -> torch.Tensor:
        size = value.shape[axis]
        if size > target:
            indices = torch.linspace(0, size - 1, target).round().long()
            value = torch.index_select(value, axis, indices)
        elif size < target:
            pad_shape = list(value.shape)
            pad_shape[axis] = target - size
            value = torch.cat([value, torch.zeros(pad_shape, dtype=value.dtype)], dim=axis)
        return value

    @staticmethod
    def _resize_nearest_2d(value: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Resize ``[..., H, W]`` without MUSA's incompatible interpolate wrapper.

        The MUSA PyTorch build used on the Moore Threads nodes rejects the
        ``scale_factors=None`` argument that ``F.interpolate(..., mode="nearest")``
        passes internally.  Cache tensors normally already have the canonical
        shape, so keep that path allocation-free and use index selection only
        when a legacy cache needs normalization.
        """

        target_h, target_w = (int(target_size[0]), int(target_size[1]))
        input_h, input_w = value.shape[-2:]
        if (input_h, input_w) == (target_h, target_w):
            return value
        y_indices = torch.linspace(0, input_h - 1, target_h).round().long()
        x_indices = torch.linspace(0, input_w - 1, target_w).round().long()
        return value.index_select(-2, y_indices).index_select(-1, x_indices)

    @staticmethod
    def _resize_bilinear_2d(value: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Pure-tensor bilinear resize for ``[C, H, W]`` cache tensors."""

        target_h, target_w = (int(target_size[0]), int(target_size[1]))
        input_h, input_w = value.shape[-2:]
        if (input_h, input_w) == (target_h, target_w):
            return value

        y = ((torch.arange(target_h, dtype=torch.float32) + 0.5) * input_h / target_h - 0.5).clamp(
            0, input_h - 1
        )
        y0 = y.floor().long()
        y1 = (y0 + 1).clamp(max=input_h - 1)
        wy = (y - y0).to(dtype=value.dtype).view(1, target_h, 1)
        resized_h = value.index_select(-2, y0) * (1 - wy) + value.index_select(-2, y1) * wy

        x = ((torch.arange(target_w, dtype=torch.float32) + 0.5) * input_w / target_w - 0.5).clamp(
            0, input_w - 1
        )
        x0 = x.floor().long()
        x1 = (x0 + 1).clamp(max=input_w - 1)
        wx = (x - x0).to(dtype=value.dtype).view(1, 1, target_w)
        return resized_h.index_select(-1, x0) * (1 - wx) + resized_h.index_select(-1, x1) * wx

    def _empty_condition(self) -> dict[str, torch.Tensor]:
        return {
            # Keep a tiny sentinel so the existing conditioner schema remains
            # compatible without allocating or transferring teacher features.
            "sam3d_tokens_B_F_N_D": torch.zeros(1, 1, self.sam3d_token_dim, dtype=torch.float32),
            "sam_mask_B_K_H_W": torch.zeros(
                self.sam_mask_instances, *self.sam_mask_size, dtype=torch.float32
            ),
            "sam_mask_meta_B_K_D": torch.zeros(
                self.sam_mask_instances, self.sam_mask_meta_dim, dtype=torch.float32
            ),
            "sam3d_geometry_B_C_H_W": torch.zeros(
                self.sam3d_geometry_channels, *self.sam3d_geometry_size, dtype=torch.float32
            ),
            "sam3d_shape_latents_B_K_N_D": torch.zeros(
                self.sam3d_shape_instances, self.sam3d_shape_tokens, self.sam3d_shape_dim, dtype=torch.float32
            ),
            "sam3d_object_pose_B_K_D": torch.zeros(
                self.sam3d_shape_instances, self.sam3d_pose_dim, dtype=torch.float32
            ),
        }

    def _load_condition(self, index: int) -> dict[str, torch.Tensor]:
        path = self.sam3d_condition_paths[index]
        if not os.path.isfile(path):
            if self.sam3d_required:
                raise FileNotFoundError(path)
            return self._empty_condition()

        cache = self._load_tensor_cache(path)
        object_path = self.sam3d_object_paths[index]
        if self.sam3d_require_complete_conditions and (object_path is None or not os.path.isfile(object_path)):
            raise FileNotFoundError(f"Required SAM 3D object sidecar disappeared after filtering: {object_path}")
        if object_path is not None and os.path.isfile(object_path):
            native = self._load_tensor_cache(object_path)
            cache = dict(cache)
            for key in ("sam3d_geometry", "sam3d_shape_latents", "sam3d_object_pose"):
                if self.sam3d_require_complete_conditions and key not in native:
                    raise KeyError(f"Required {key} is missing from {object_path}")
                cache[key] = native[key]
        if self.sam3d_require_complete_conditions:
            for key in ("sam_masks", "sam_mask_meta", "sam3d_geometry", "sam3d_shape_latents", "sam3d_object_pose"):
                if key not in cache:
                    raise KeyError(f"Required {key} is missing from {path}")
        if self.sam3d_load_teacher_tokens:
            tokens = torch.as_tensor(cache["sam3d_tokens"], dtype=torch.float32)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 3 or tokens.shape[-1] != self.sam3d_token_dim:
                raise ValueError(
                    f"sam3d_tokens in {path} must be [F,N,{self.sam3d_token_dim}], got {tuple(tokens.shape)}"
                )
            tokens = self._select_or_pad_axis(tokens, self.sam3d_teacher_frames, axis=0)
            tokens = self._select_or_pad_axis(tokens, self.sam3d_num_tokens, axis=1)
        else:
            tokens = torch.zeros(1, 1, self.sam3d_token_dim, dtype=torch.float32)

        masks = torch.as_tensor(cache["sam_masks"], dtype=torch.float32)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        masks = self._select_or_pad_axis(masks, self.sam_mask_instances, axis=0)
        masks = self._resize_nearest_2d(masks, self.sam_mask_size)

        mask_meta = torch.as_tensor(cache["sam_mask_meta"], dtype=torch.float32)
        if mask_meta.ndim == 1:
            mask_meta = mask_meta.unsqueeze(0)
        mask_meta = self._select_or_pad_axis(mask_meta, self.sam_mask_instances, axis=0)
        mask_meta = self._select_or_pad_axis(mask_meta, self.sam_mask_meta_dim, axis=1)

        geometry = torch.as_tensor(cache["sam3d_geometry"], dtype=torch.float32)
        if geometry.ndim == 2:
            geometry = geometry.unsqueeze(0)
        geometry = self._select_or_pad_axis(geometry, self.sam3d_geometry_channels, axis=0)
        geometry = self._resize_bilinear_2d(geometry, self.sam3d_geometry_size)

        shape_latents = torch.as_tensor(
            cache.get(
                "sam3d_shape_latents",
                torch.zeros(self.sam3d_shape_instances, self.sam3d_shape_tokens, self.sam3d_shape_dim),
            ),
            dtype=torch.float32,
        )
        if shape_latents.ndim == 2:
            shape_latents = shape_latents.unsqueeze(0)
        if shape_latents.ndim != 3 or shape_latents.shape[-1] != self.sam3d_shape_dim:
            raise ValueError(
                f"sam3d_shape_latents in {path} must be [K,N,{self.sam3d_shape_dim}], "
                f"got {tuple(shape_latents.shape)}"
            )
        shape_latents = self._select_or_pad_axis(shape_latents, self.sam3d_shape_instances, axis=0)
        shape_latents = self._select_or_pad_axis(shape_latents, self.sam3d_shape_tokens, axis=1)

        object_pose = torch.as_tensor(
            cache.get("sam3d_object_pose", torch.zeros(self.sam3d_shape_instances, self.sam3d_pose_dim)),
            dtype=torch.float32,
        )
        if object_pose.ndim == 1:
            object_pose = object_pose.unsqueeze(0)
        if object_pose.ndim != 2:
            raise ValueError(f"sam3d_object_pose in {path} must be [K,D], got {tuple(object_pose.shape)}")
        object_pose = self._select_or_pad_axis(object_pose, self.sam3d_shape_instances, axis=0)
        object_pose = self._select_or_pad_axis(object_pose, self.sam3d_pose_dim, axis=1)

        return {
            "sam3d_tokens_B_F_N_D": tokens.contiguous(),
            "sam_mask_B_K_H_W": masks.contiguous(),
            "sam_mask_meta_B_K_D": mask_meta.contiguous(),
            "sam3d_geometry_B_C_H_W": geometry.contiguous(),
            "sam3d_shape_latents_B_K_N_D": shape_latents.contiguous(),
            "sam3d_object_pose_B_K_D": object_pose.contiguous(),
        }

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data: dict[str, Any] = {}
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = (
                    self.caption_paths[index]
                    if self.caption_paths is not None
                    else os.path.join(self.caption_dir, f"{video_basename}.json")
                )
                caption = self._load_json_caption(Path(caption_path))
            else:
                caption = self._load_text(Path(os.path.join(self.caption_dir, f"{video_basename}.txt")))

            data["video"] = video
            data["ai_caption"] = caption
            _, _, height, width = video.shape
            data["fps"] = fps
            data["image_size"] = torch.tensor([height, width, height, width])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, height, width)
            data.update(self._load_condition(index))
            return data
        except Exception as error:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load SAM 3D video sample {self.video_paths[index]} "
                f"(total failures: {self.num_failed_loads}): {error}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            # With a one-sample smoke dataset recursive retrying can never
            # recover and previously spawned an unbounded failure loop.
            if len(self.video_paths) <= 1:
                raise
            return self[np.random.randint(len(self.video_paths))]
