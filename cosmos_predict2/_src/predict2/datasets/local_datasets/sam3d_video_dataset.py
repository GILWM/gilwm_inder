# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Video dataset that attaches fixed-shape SAM 3 / SAM 3D cache tensors."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset
from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import (
    ACTION_DIM,
    ActionNormStats,
    WorldArenaActionIndex,
    read_action_sequence,
    resample_action_sequence,
)


class SAM3DVideoDataset(VideoDataset):
    """Load ordinary Cosmos samples plus offline frozen-teacher features.

    Cache layout mirrors the compact training-data layout::

        CACHE_ROOT/<batch>/<question_id>/condition.pt

    Each cache contains ``sam3d_tokens``, ``sam_masks``, ``sam_mask_meta``
    and ``sam3d_geometry``.  Shapes are normalized here so the default
    DataLoader collate function remains deterministic.
    """

    @staticmethod
    def _parse_bool(value: bool | str, name: str) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"{name} must be true or false, got {value!r}")
            return normalized == "true"
        return bool(value)

    def __init__(
        self,
        *args,
        sam3d_cache_dir: str,
        sam3d_required: bool = True,
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
        action_hdf5_root: Optional[str] = None,
        action_required: bool = False,
        action_norm_path: Optional[str] = None,
        action_alignment_offset: int = 0,
        action_validate_vector: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sam3d_cache_dir = sam3d_cache_dir
        self.sam3d_required = sam3d_required
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
        self.action_hdf5_root = action_hdf5_root
        self.action_required = self._parse_bool(action_required, "action_required")
        self.action_alignment_offset = int(action_alignment_offset)
        self.action_validate_vector = self._parse_bool(action_validate_vector, "action_validate_vector")
        self.action_norm = ActionNormStats.load(action_norm_path) if action_norm_path else None
        manifest_action_overrides = self._manifest_action_overrides()
        self.action_enabled = self.action_hdf5_root is not None or bool(manifest_action_overrides)
        has_absolute_manifest_actions = any(os.path.isabs(path) for path in manifest_action_overrides.values())
        if self.action_required and self.action_hdf5_root is None and not has_absolute_manifest_actions:
            raise ValueError(
                "action_required=True requires action_hdf5_root or explicit action paths in manifest_paths"
            )
        if repeat_factor < 1:
            raise ValueError(f"repeat_factor must be >= 1, got {repeat_factor}")

        self.sam3d_condition_paths = [self._condition_path(video_path) for video_path in self.video_paths]
        if self.sam3d_required:
            keep_indices = [index for index, path in enumerate(self.sam3d_condition_paths) if os.path.isfile(path)]
            missing_count = len(self.video_paths) - len(keep_indices)
            self.video_paths = [self.video_paths[index] for index in keep_indices]
            self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
            if self.caption_paths is not None:
                self.caption_paths = [self.caption_paths[index] for index in keep_indices]
            log.info(
                f"SAM 3D cache coverage: kept {len(keep_indices)} samples and excluded {missing_count} missing caches"
            )
            if not self.video_paths:
                raise ValueError(f"No SAM 3D caches found under {self.sam3d_cache_dir}")

        if sam3d_native_index is not None:
            index_path = Path(sam3d_native_index)
            if not index_path.is_file():
                raise FileNotFoundError(index_path)
            native_samples = {
                line.strip().replace("\\", "/")
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            keep_indices = []
            for index, condition_path in enumerate(self.sam3d_condition_paths):
                sample = os.path.relpath(os.path.dirname(condition_path), self.sam3d_cache_dir).replace("\\", "/")
                if sample in native_samples:
                    keep_indices.append(index)
            self.video_paths = [self.video_paths[index] for index in keep_indices]
            self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
            if self.caption_paths is not None:
                self.caption_paths = [self.caption_paths[index] for index in keep_indices]
            log.info(
                f"Native SAM 3D index: kept {len(keep_indices)} samples from {len(native_samples)} indexed sidecars"
            )
            if not self.video_paths:
                raise ValueError(f"No dataset samples matched native SAM 3D index {index_path}")

        self.sam3d_object_paths = [self._object_path(path) for path in self.sam3d_condition_paths]
        if self.sam3d_objects_root is not None:
            native_count = sum(os.path.isfile(path) for path in self.sam3d_object_paths)
            log.info(
                f"Native SAM 3D sidecar overlay: found {native_count} of "
                f"{len(self.sam3d_object_paths)} dataset samples under {self.sam3d_objects_root}"
            )

        self.action_hdf5_paths: list[Optional[str]] = [None] * len(self.video_paths)
        if self.action_hdf5_root is not None or manifest_action_overrides:
            action_index = WorldArenaActionIndex(self.action_hdf5_root) if self.action_hdf5_root is not None else None
            self.action_hdf5_paths = [
                self._action_path(video_path, action_index, manifest_action_overrides)
                for video_path in self.video_paths
            ]
            action_count = sum(path is not None and os.path.isfile(path) for path in self.action_hdf5_paths)
            log.info(
                f"WorldArena action sidecars: found {action_count} of "
                f"{len(self.video_paths)} samples from explicit manifests/root {self.action_hdf5_root}"
            )
            if self.action_required:
                keep_indices = [
                    index
                    for index, path in enumerate(self.action_hdf5_paths)
                    if path is not None and os.path.isfile(path)
                ]
                missing_count = len(self.video_paths) - len(keep_indices)
                self.video_paths = [self.video_paths[index] for index in keep_indices]
                self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
                self.sam3d_object_paths = [self.sam3d_object_paths[index] for index in keep_indices]
                self.action_hdf5_paths = [self.action_hdf5_paths[index] for index in keep_indices]
                if self.caption_paths is not None:
                    self.caption_paths = [self.caption_paths[index] for index in keep_indices]
                log.info(
                    f"Required action coverage: kept {len(keep_indices)} samples and "
                    f"excluded {missing_count} missing HDF5 sidecars"
                )
                if not self.video_paths:
                    raise ValueError(
                        f"No video samples could be paired with HDF5 actions under {self.action_hdf5_root}. "
                        "The public WorldArena Track 1 HDF5 files do not contain RGB video; "
                        "provide a paired video dataset with matching episode IDs or manifest paths."
                    )

        # Small-scale memory/stability tests deliberately use a handful of
        # fully native sidecars.  Logical repetition supplies enough local
        # samples for large per-device batches without copying cache files or
        # changing full-scale sampling, where repeat_factor remains one.
        if repeat_factor > 1:
            self.video_paths = self.video_paths * repeat_factor
            self.sam3d_condition_paths = self.sam3d_condition_paths * repeat_factor
            self.sam3d_object_paths = self.sam3d_object_paths * repeat_factor
            self.action_hdf5_paths = self.action_hdf5_paths * repeat_factor
            if self.caption_paths is not None:
                self.caption_paths = self.caption_paths * repeat_factor
            log.info(f"Repeated SAM 3D dataset {repeat_factor}x for {len(self.video_paths)} logical samples")

    def _condition_path(self, video_path: str) -> str:
        relative_video_path = os.path.relpath(video_path, self.dataset_dir)
        return os.path.join(self.sam3d_cache_dir, os.path.dirname(relative_video_path), "condition.pt")

    def _object_path(self, condition_path: str) -> Optional[str]:
        if self.sam3d_objects_root is None:
            return None
        sample = os.path.relpath(os.path.dirname(condition_path), self.sam3d_cache_dir)
        return os.path.join(self.sam3d_objects_root, sample, "sam3d_objects.pt")

    def _manifest_action_overrides(self) -> dict[str, str]:
        """Read explicit video -> HDF5 pairs from the compact dataset manifest."""

        result: dict[str, str] = {}
        manifests = getattr(self, "manifest_paths", None) or [Path(self.dataset_dir) / "manifest.jsonl"]
        explicit_manifests = bool(getattr(self, "_explicit_manifest_paths", False))
        for manifest_value in manifests:
            manifest = Path(manifest_value)
            if not manifest.is_file():
                continue
            for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                video = record.get("video")
                action = next(
                    (
                        record.get(key)
                        for key in ("trajectory_hdf5", "action_hdf5", "hdf5", "actions")
                        if record.get(key)
                    ),
                    None,
                )
                if action is None:
                    continue
                if not isinstance(video, str) or not isinstance(action, str):
                    raise ValueError(f"{manifest}:{line_number}: invalid video/action path fields")
                video_path = os.path.normpath(os.path.join(manifest.parent, video))
                if explicit_manifests and not os.path.isabs(action):
                    action = os.path.normpath(os.path.join(manifest.parent, action))
                result[video_path] = action
        return result

    def _action_path(
        self,
        video_path: str,
        action_index: Optional[WorldArenaActionIndex],
        manifest_overrides: dict[str, str],
    ) -> Optional[str]:
        override = manifest_overrides.get(os.path.normpath(video_path))
        if override is not None:
            candidate = Path(override)
            if not candidate.is_absolute():
                if self.action_hdf5_root is None:
                    return None
                candidate = Path(self.action_hdf5_root) / candidate
            return os.path.abspath(candidate)
        if action_index is None:
            return None
        resolved = action_index.resolve(video_path, dataset_root=self.dataset_dir)
        return str(resolved) if resolved is not None else None

    def _load_actions(
        self,
        index: int,
        video_frame_indices: np.ndarray,
        video_frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.action_hdf5_paths[index]
        if path is None or not os.path.isfile(path):
            if self.action_required:
                raise FileNotFoundError(path or f"No action sidecar for {self.video_paths[index]}")
            return (
                torch.zeros(self.sequence_length, ACTION_DIM, dtype=torch.float32),
                torch.tensor(False, dtype=torch.bool),
            )
        actions = read_action_sequence(path, validate_vector=self.action_validate_vector)
        actions, _ = resample_action_sequence(
            actions,
            target_frames=self.sequence_length,
            video_frame_indices=video_frame_indices,
            video_frame_count=video_frame_count,
            alignment_offset=self.action_alignment_offset,
        )
        if self.action_norm is not None:
            actions = self.action_norm.normalize(actions)
        return torch.from_numpy(actions.copy()), torch.tensor(True, dtype=torch.bool)

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

        y = ((torch.arange(target_h, dtype=torch.float32) + 0.5) * input_h / target_h - 0.5).clamp(0, input_h - 1)
        y0 = y.floor().long()
        y1 = (y0 + 1).clamp(max=input_h - 1)
        wy = (y - y0).to(dtype=value.dtype).view(1, target_h, 1)
        resized_h = value.index_select(-2, y0) * (1 - wy) + value.index_select(-2, y1) * wy

        x = ((torch.arange(target_w, dtype=torch.float32) + 0.5) * input_w / target_w - 0.5).clamp(0, input_w - 1)
        x0 = x.floor().long()
        x1 = (x0 + 1).clamp(max=input_w - 1)
        wx = (x - x0).to(dtype=value.dtype).view(1, 1, target_w)
        return resized_h.index_select(-1, x0) * (1 - wx) + resized_h.index_select(-1, x1) * wx

    def _empty_condition(self) -> dict[str, torch.Tensor]:
        return {
            "sam3d_tokens_B_F_N_D": torch.zeros(
                self.sam3d_teacher_frames, self.sam3d_num_tokens, self.sam3d_token_dim, dtype=torch.float32
            ),
            "sam_mask_B_K_H_W": torch.zeros(self.sam_mask_instances, *self.sam_mask_size, dtype=torch.float32),
            "sam_mask_meta_B_K_D": torch.zeros(self.sam_mask_instances, self.sam_mask_meta_dim, dtype=torch.float32),
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
        if object_path is not None and os.path.isfile(object_path):
            native = self._load_tensor_cache(object_path)
            cache = dict(cache)
            cache["sam3d_geometry"] = native["sam3d_geometry"]
            cache["sam3d_shape_latents"] = native["sam3d_shape_latents"]
            cache["sam3d_object_pose"] = native["sam3d_object_pose"]
        tokens = torch.as_tensor(cache["sam3d_tokens"], dtype=torch.float32)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3 or tokens.shape[-1] != self.sam3d_token_dim:
            raise ValueError(f"sam3d_tokens in {path} must be [F,N,{self.sam3d_token_dim}], got {tuple(tokens.shape)}")
        tokens = self._select_or_pad_axis(tokens, self.sam3d_teacher_frames, axis=0)
        tokens = self._select_or_pad_axis(tokens, self.sam3d_num_tokens, axis=1)

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
                f"sam3d_shape_latents in {path} must be [K,N,{self.sam3d_shape_dim}], got {tuple(shape_latents.shape)}"
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
            video, fps, video_frame_indices, video_frame_count = self._get_frames_with_frame_ids(
                self.video_paths[index]
            )
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
            if self.action_enabled:
                actions, action_valid = self._load_actions(index, video_frame_indices, video_frame_count)
                data["actions_B_T_D"] = actions
                data["action_valid_B"] = action_valid
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
