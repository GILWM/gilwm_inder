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


_LOGGED_ELIGIBILITY_SUMMARIES: set[tuple[str, str, str]] = set()


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
        sam3d_load_teacher_tokens: bool = True,
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
        action_hdf5_root: Optional[str] = None,
        action_required: bool = False,
        action_norm_path: Optional[str] = None,
        action_alignment_offset: int = 0,
        action_validate_vector: bool = True,
        sam3_video_teacher_dir: Optional[str] = None,
        sam3_video_teacher_required: bool = False,
        sam3_video_mask_instances: int = 8,
        sam3_video_mask_size: tuple[int, int] = (30, 40),
        sam3_video_frames_per_latent: int = 4,
        expected_num_samples: Optional[int] = None,
        continuous_condition_policy: str = "restrict",
        max_load_retries: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sam3d_cache_dir = sam3d_cache_dir
        self.sam3d_required = self._parse_bool(sam3d_required, "sam3d_required")
        self.sam3d_load_teacher_tokens = self._parse_bool(
            sam3d_load_teacher_tokens, "sam3d_load_teacher_tokens"
        )
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
        self.sam3d_require_complete_conditions = self._parse_bool(
            sam3d_require_complete_conditions, "sam3d_require_complete_conditions"
        )
        self.action_hdf5_root = action_hdf5_root
        self.action_required = self._parse_bool(action_required, "action_required")
        self.action_alignment_offset = int(action_alignment_offset)
        self.action_validate_vector = self._parse_bool(action_validate_vector, "action_validate_vector")
        self.action_norm = ActionNormStats.load(action_norm_path) if action_norm_path else None
        self.sam3_video_teacher_dir = sam3_video_teacher_dir
        self.sam3_video_teacher_required = self._parse_bool(
            sam3_video_teacher_required, "sam3_video_teacher_required"
        )
        self.sam3_video_mask_instances = int(sam3_video_mask_instances)
        self.sam3_video_mask_size = tuple(int(value) for value in sam3_video_mask_size)
        self.sam3_video_frames_per_latent = int(sam3_video_frames_per_latent)
        if continuous_condition_policy not in {"restrict", "drop_unaligned"}:
            raise ValueError(
                "continuous_condition_policy must be restrict or drop_unaligned, "
                f"got {continuous_condition_policy!r}"
            )
        self.continuous_condition_policy = continuous_condition_policy
        self.max_load_retries = int(max_load_retries)
        if self.max_load_retries < 0:
            raise ValueError(f"max_load_retries must be non-negative, got {max_load_retries}")
        self.expected_num_samples = int(expected_num_samples) if expected_num_samples is not None else None
        if self.expected_num_samples is not None and self.expected_num_samples < 1:
            raise ValueError(f"expected_num_samples must be positive, got {expected_num_samples}")
        if min(self.sam3_video_mask_instances, self.sam3_video_frames_per_latent, *self.sam3_video_mask_size) <= 0:
            raise ValueError("SAM 3 video teacher dimensions must be positive")
        manifest_action_overrides = self._manifest_action_overrides()
        self.action_enabled = self.action_hdf5_root is not None or bool(manifest_action_overrides)
        has_absolute_manifest_actions = any(os.path.isabs(path) for path in manifest_action_overrides.values())
        if self.action_required and self.action_hdf5_root is None and not has_absolute_manifest_actions:
            raise ValueError(
                "action_required=True requires action_hdf5_root or explicit action paths in manifest_paths"
            )
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
                raise ValueError("sam3d_require_complete_conditions=True requires sam3d_objects_root")

        candidate_count = len(self.video_paths)
        missing_cache_count = 0
        manifest_condition_samples = self._manifest_condition_samples()
        self.sam3d_condition_paths = [
            self._condition_path(video_path, manifest_condition_samples) for video_path in self.video_paths
        ]
        if self.sam3d_required:
            keep_indices = [index for index, path in enumerate(self.sam3d_condition_paths) if os.path.isfile(path)]
            missing_cache_count = len(self.video_paths) - len(keep_indices)
            self.video_paths = [self.video_paths[index] for index in keep_indices]
            self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
            if self.caption_paths is not None:
                self.caption_paths = [self.caption_paths[index] for index in keep_indices]
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
        missing_sidecar_count = 0
        if self.sam3d_require_complete_conditions:
            keep_indices = [
                index
                for index, path in enumerate(self.sam3d_object_paths)
                if path is not None and os.path.isfile(path)
            ]
            missing_sidecar_count = len(self.video_paths) - len(keep_indices)
            self.video_paths = [self.video_paths[index] for index in keep_indices]
            self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
            self.sam3d_object_paths = [self.sam3d_object_paths[index] for index in keep_indices]
            if self.caption_paths is not None:
                self.caption_paths = [self.caption_paths[index] for index in keep_indices]
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
                    f"validated_index_entries={native_sample_count} "
                    f"teacher_tokens_required={str(self.sam3d_load_teacher_tokens).lower()}",
                    rank0_only=True,
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

        self.sam3_video_teacher_paths: list[Optional[str]] = [
            self._video_teacher_path(path) if self.sam3_video_teacher_dir is not None else None
            for path in self.sam3d_condition_paths
        ]
        if self.sam3_video_teacher_required:
            if self.sam3_video_teacher_dir is None:
                raise ValueError("sam3_video_teacher_required=True requires sam3_video_teacher_dir")
            keep_indices = [
                index for index, path in enumerate(self.sam3_video_teacher_paths) if path is not None and os.path.isfile(path)
            ]
            missing_count = len(self.video_paths) - len(keep_indices)
            self.video_paths = [self.video_paths[index] for index in keep_indices]
            self.sam3d_condition_paths = [self.sam3d_condition_paths[index] for index in keep_indices]
            self.sam3d_object_paths = [self.sam3d_object_paths[index] for index in keep_indices]
            self.action_hdf5_paths = [self.action_hdf5_paths[index] for index in keep_indices]
            self.sam3_video_teacher_paths = [self.sam3_video_teacher_paths[index] for index in keep_indices]
            if self.caption_paths is not None:
                self.caption_paths = [self.caption_paths[index] for index in keep_indices]
            log.info(
                f"SAM 3 video teacher coverage: kept {len(keep_indices)} samples and "
                f"excluded {missing_count} missing caches"
            )
            if not self.video_paths:
                raise ValueError(f"No SAM 3 video teacher caches found under {self.sam3_video_teacher_dir}")

        # Repeat only after every required sidecar filter.  This keeps smoke
        # tests with one teacher cache from stat'ing hundreds of thousands of
        # duplicated missing paths and preserves identical formal behavior at
        # the default repeat_factor=1.
        if repeat_factor > 1:
            self.video_paths = self.video_paths * repeat_factor
            self.sam3d_condition_paths = self.sam3d_condition_paths * repeat_factor
            self.sam3d_object_paths = self.sam3d_object_paths * repeat_factor
            self.action_hdf5_paths = self.action_hdf5_paths * repeat_factor
            self.sam3_video_teacher_paths = self.sam3_video_teacher_paths * repeat_factor
            if self.caption_paths is not None:
                self.caption_paths = self.caption_paths * repeat_factor
            log.info(f"Repeated SAM 3D dataset {repeat_factor}x for {len(self.video_paths)} logical samples")

        if self.expected_num_samples is not None and len(self.video_paths) != self.expected_num_samples:
            raise RuntimeError(
                "Distributed dataset integrity check failed: "
                f"expected {self.expected_num_samples} usable samples, found {len(self.video_paths)}. "
                "Check manifest filters, action sidecars, video frame-count cache, and container mounts on this node."
            )

    def _condition_path(self, video_path: str, manifest_overrides: Optional[dict[str, str]] = None) -> str:
        override = (manifest_overrides or {}).get(os.path.normpath(video_path))
        if override is not None:
            return os.path.join(self.sam3d_cache_dir, override, "condition.pt")
        relative_video_path = os.path.relpath(video_path, self.dataset_dir)
        return os.path.join(self.sam3d_cache_dir, os.path.dirname(relative_video_path), "condition.pt")

    def _manifest_condition_samples(self) -> dict[str, str]:
        """Map filtered/symlinked views back to canonical cache sample IDs."""

        result: dict[str, str] = {}
        for manifest_value in getattr(self, "manifest_paths", None) or []:
            manifest = Path(manifest_value)
            if not manifest.is_file():
                continue
            for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                video = record.get("video")
                if not isinstance(video, str):
                    raise ValueError(f"{manifest}:{line_number}: invalid video path")
                video_path = os.path.normpath(os.path.join(manifest.parent, video))
                sample = record.get("source_sample_dir") or record.get("sample_dir")
                if not sample and record.get("batch") and record.get("question_id"):
                    sample = f"{record['batch']}/{record['question_id']}"
                if sample:
                    result[video_path] = os.path.normpath(str(sample))
        return result

    def _object_path(self, condition_path: str) -> Optional[str]:
        if self.sam3d_objects_root is None:
            return None
        sample = os.path.relpath(os.path.dirname(condition_path), self.sam3d_cache_dir)
        return os.path.join(self.sam3d_objects_root, sample, "sam3d_objects.pt")

    def _video_teacher_path(self, condition_path: str) -> str:
        sample = os.path.relpath(os.path.dirname(condition_path), self.sam3d_cache_dir)
        return os.path.join(str(self.sam3_video_teacher_dir), sample, "sam3_video_teacher.pt")

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

    def _load_sam3_video_teacher(
        self,
        index: int,
        video_frame_indices: np.ndarray,
    ) -> Optional[dict[str, torch.Tensor]]:
        path = self.sam3_video_teacher_paths[index]
        if path is None or not os.path.isfile(path):
            if self.sam3_video_teacher_required:
                raise FileNotFoundError(path or f"No SAM 3 video teacher for {self.video_paths[index]}")
            return None
        cache = self._load_tensor_cache(path)
        masks = torch.as_tensor(cache["sam3_video_masks_T_K_H_W"], dtype=torch.float32)
        valid = torch.as_tensor(cache["sam3_video_valid_T_K"], dtype=torch.bool)
        source_indices = torch.as_tensor(cache["source_frame_indices"], dtype=torch.int64).reshape(-1)
        if masks.ndim != 4 or valid.ndim != 2:
            raise ValueError(f"Malformed SAM 3 video teacher shapes in {path}: {masks.shape}, {valid.shape}")
        if masks.shape[:2] != valid.shape or masks.shape[0] != len(source_indices):
            raise ValueError(f"SAM 3 video teacher timeline mismatch in {path}")

        # Cosmos's causal 4x temporal VAE maps RGB indices 0,4,... to its
        # latent timeline.  Index the exact sampled source frames; never use a
        # nearby source mask because that would silently misalign motion.
        target_source_indices = np.asarray(video_frame_indices, dtype=np.int64)[:: self.sam3_video_frames_per_latent]
        lookup = {int(source): position for position, source in enumerate(source_indices.tolist())}
        missing = [int(source) for source in target_source_indices if int(source) not in lookup]
        if missing:
            raise ValueError(
                f"SAM 3 video teacher {path} lacks sampled source frames {missing[:8]}"
                + ("..." if len(missing) > 8 else "")
            )
        positions = torch.tensor([lookup[int(source)] for source in target_source_indices], dtype=torch.long)
        masks = masks.index_select(0, positions)
        valid = valid.index_select(0, positions)
        masks = self._select_or_pad_axis(masks, self.sam3_video_mask_instances, axis=1)
        valid = self._select_or_pad_axis(valid, self.sam3_video_mask_instances, axis=1)
        masks = self._resize_nearest_2d(masks, self.sam3_video_mask_size)
        return {
            "sam3_video_masks_B_T_K_H_W": masks.contiguous(),
            "sam3_video_valid_B_T_K": valid.contiguous(),
        }

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
        token_shape = (
            (self.sam3d_teacher_frames, self.sam3d_num_tokens, self.sam3d_token_dim)
            if self.sam3d_load_teacher_tokens
            else (1, 1, self.sam3d_token_dim)
        )
        return {
            "sam3d_tokens_B_F_N_D": torch.zeros(*token_shape, dtype=torch.float32),
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

    def _load_raw_condition(self, index: int) -> Optional[dict[str, Any]]:
        path = self.sam3d_condition_paths[index]
        if not os.path.isfile(path):
            if self.sam3d_required:
                raise FileNotFoundError(path)
            return None

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
            # A native sidecar may contain several crop-start anchors.  Keep
            # its timeline separate from the SAM-mask timeline in the base
            # cache so one cannot silently index the other by position.
            native_indices = native.get("condition_frame_indices")
            for field in ("sam3d_geometry", "sam3d_shape_latents", "sam3d_object_pose"):
                field_indices = native.get(f"{field}_frame_indices", native_indices)
                if field_indices is not None:
                    cache[f"{field}_frame_indices"] = field_indices
            if "geometry_valid" in native:
                cache["geometry_valid"] = native["geometry_valid"]
            if "sam3d_object_valid_count" in native:
                cache["sam3d_object_valid_count"] = native["sam3d_object_valid_count"]
            elif "object_count" in native:
                cache["sam3d_object_valid_count"] = native["object_count"]
        if self.sam3d_require_complete_conditions:
            for key in (
                "sam_masks",
                "sam_mask_meta",
                "sam3d_geometry",
                "sam3d_shape_latents",
                "sam3d_object_pose",
            ):
                if key not in cache:
                    raise KeyError(f"Required {key} is missing from complete condition sample {path}")
        return cache

    @staticmethod
    def _field_frame_indices(
        cache: dict[str, Any],
        field: str,
        value: torch.Tensor,
        legacy_ndim: int,
    ) -> np.ndarray:
        """Return source-frame anchors for a condition tensor.

        Schema-v2 caches have no temporal axis and were always extracted from
        source frame zero.  Schema-v3 tensors add a leading anchor axis and
        must carry exact source-frame indices.
        """

        if value.ndim == legacy_ndim:
            index = cache.get(f"{field}_frame_index", cache.get("condition_frame_index", 0))
            return np.asarray([int(torch.as_tensor(index).item())], dtype=np.int64)
        if value.ndim != legacy_ndim + 1:
            raise ValueError(
                f"{field} must have {legacy_ndim} (legacy) or {legacy_ndim + 1} (anchored) dimensions, "
                f"got {tuple(value.shape)}"
            )
        raw_indices = cache.get(f"{field}_frame_indices", cache.get("condition_frame_indices"))
        if raw_indices is None:
            raise ValueError(f"Anchored {field} is missing exact source-frame indices")
        indices = torch.as_tensor(raw_indices, dtype=torch.int64).reshape(-1).cpu().numpy()
        if len(indices) != value.shape[0]:
            raise ValueError(
                f"{field} has {value.shape[0]} anchors but {len(indices)} source-frame indices"
            )
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{field} source-frame indices must be unique, got {indices.tolist()}")
        return indices

    @classmethod
    def _select_condition_frame(
        cls,
        cache: dict[str, Any],
        field: str,
        value: torch.Tensor,
        legacy_ndim: int,
        source_frame_index: int,
    ) -> torch.Tensor:
        indices = cls._field_frame_indices(cache, field, value, legacy_ndim)
        matches = np.flatnonzero(indices == int(source_frame_index))
        if not len(matches):
            raise ValueError(
                f"{field} has no condition for sampled first frame {source_frame_index}; "
                f"available anchors are {indices.tolist()}"
            )
        if value.ndim == legacy_ndim:
            return value
        return value[int(matches[0])]

    @staticmethod
    def _has_active_values(value: Any) -> bool:
        tensor = torch.as_tensor(value)
        return bool(tensor.numel() and torch.count_nonzero(tensor).item())

    def _condition_start_frames(self, cache: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
        """Find crop starts for which every active first-frame modality aligns.

        Continuous crops are sampled only from this intersection.  Therefore
        an old frame-zero cache remains correct (it temporarily restricts the
        crop start to zero) instead of being attached to an unrelated random
        crop.  Regenerated schema-v3 caches restore random starts across their
        exact cached anchors.
        """

        if cache is None:
            return None

        fields = (
            ("sam_masks", 3, True),
            ("sam_mask_meta", 2, True),
            ("sam3d_geometry", 3, self._has_active_values(cache.get("sam3d_geometry", []))),
            ("sam3d_shape_latents", 3, self._has_active_values(cache.get("sam3d_shape_latents", []))),
            ("sam3d_object_pose", 2, self._has_active_values(cache.get("sam3d_object_pose", []))),
        )
        aligned: Optional[set[int]] = None
        for field, legacy_ndim, active in fields:
            if not active or field not in cache:
                continue
            value = torch.as_tensor(cache[field])
            indices = set(self._field_frame_indices(cache, field, value, legacy_ndim).tolist())
            aligned = indices if aligned is None else aligned.intersection(indices)
        if not aligned:
            raise ValueError("No common source-frame anchor exists across active SAM 3 / SAM 3D conditions")
        return np.asarray(sorted(aligned), dtype=np.int64)

    def _condition_cache_for_sampled_frame(
        self,
        cache: Optional[dict[str, Any]],
        source_frame_index: int,
    ) -> Optional[dict[str, Any]]:
        """Return only a condition cache that exactly matches the sampled first frame.

        ``restrict`` preserves the historical behavior: a continuous crop is
        selected from cached anchors before this method is called.  The
        action-focused ``drop_unaligned`` policy instead samples the video
        freely and converts a legacy/missing anchor into a real unconditional
        SAM/SAM3D sample.  It never attaches frame-zero structure to a random
        later crop.
        """

        if cache is None or self.continuous_condition_policy == "restrict":
            return cache
        anchors = self._condition_start_frames(cache)
        if anchors is not None and int(source_frame_index) in set(anchors.tolist()):
            return cache
        return None

    def _load_condition(
        self,
        index: int,
        source_frame_index: int = 0,
        cache: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        path = self.sam3d_condition_paths[index]
        if cache is None:
            cache = self._load_raw_condition(index)
        if cache is None:
            return self._empty_condition()

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
        masks = self._select_condition_frame(cache, "sam_masks", masks, 3, source_frame_index)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        masks = self._select_or_pad_axis(masks, self.sam_mask_instances, axis=0)
        masks = self._resize_nearest_2d(masks, self.sam_mask_size)

        mask_meta = torch.as_tensor(cache["sam_mask_meta"], dtype=torch.float32)
        mask_meta = self._select_condition_frame(cache, "sam_mask_meta", mask_meta, 2, source_frame_index)
        if mask_meta.ndim == 1:
            mask_meta = mask_meta.unsqueeze(0)
        mask_meta = self._select_or_pad_axis(mask_meta, self.sam_mask_instances, axis=0)
        mask_meta = self._select_or_pad_axis(mask_meta, self.sam_mask_meta_dim, axis=1)

        geometry = torch.as_tensor(cache["sam3d_geometry"], dtype=torch.float32)
        if self._has_active_values(geometry):
            geometry = self._select_condition_frame(cache, "sam3d_geometry", geometry, 3, source_frame_index)
        elif geometry.ndim == 4:
            geometry = geometry[0]
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
        if self._has_active_values(shape_latents):
            shape_latents = self._select_condition_frame(
                cache, "sam3d_shape_latents", shape_latents, 3, source_frame_index
            )
        elif shape_latents.ndim == 4:
            shape_latents = shape_latents[0]
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
        if self._has_active_values(object_pose):
            object_pose = self._select_condition_frame(
                cache, "sam3d_object_pose", object_pose, 2, source_frame_index
            )
        elif object_pose.ndim == 3:
            object_pose = object_pose[0]
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

    def _get_item(self, index: int) -> dict[str, Any]:
        data: dict[str, Any] = {}
        condition_cache = self._load_raw_condition(index)
        condition_start_frames = None
        if (
            self.sampling_mode == "continuous"
            and self.continuous_condition_policy == "restrict"
        ):
            condition_start_frames = self._condition_start_frames(condition_cache)
        video, fps, video_frame_indices, video_frame_count = self._get_frames_with_frame_ids(
            self.video_paths[index],
            continuous_start_frames=condition_start_frames,
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
        source_frame_index = int(video_frame_indices[0])
        aligned_condition_cache = self._condition_cache_for_sampled_frame(
            condition_cache,
            source_frame_index,
        )
        if aligned_condition_cache is None:
            data.update(self._empty_condition())
        else:
            data.update(
                self._load_condition(
                    index,
                    source_frame_index=source_frame_index,
                    cache=aligned_condition_cache,
                )
            )
        if self.action_enabled:
            actions, action_valid = self._load_actions(index, video_frame_indices, video_frame_count)
            data["actions_B_T_D"] = actions
            data["action_valid_B"] = action_valid
        video_teacher = self._load_sam3_video_teacher(index, video_frame_indices)
        if video_teacher is not None:
            data.update(video_teacher)
        return data

    def __getitem__(self, index: int) -> dict | Any:
        failures: list[str] = []
        for attempt in range(self.max_load_retries + 1):
            try:
                return self._get_item(index)
            except Exception as error:
                self.num_failed_loads += 1
                failures.append(f"{self.video_paths[index]}: {error}")
                log.warning(
                    f"Failed to load SAM 3D video sample {self.video_paths[index]} "
                    f"(attempt {attempt + 1}/{self.max_load_retries + 1}, "
                    f"total failures: {self.num_failed_loads}): {error}\n{traceback.format_exc()}",
                    rank0_only=False,
                )
                if len(self.video_paths) <= 1 or attempt >= self.max_load_retries:
                    raise RuntimeError(
                        "Dataset sample loading exceeded the bounded retry budget; "
                        f"recent failures: {failures[-3:]}"
                    ) from error
                index = int(np.random.randint(len(self.video_paths)))
        raise AssertionError("unreachable")
