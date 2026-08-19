# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""WorldArena/RoboTwin HDF5 action sidecar utilities.

The public WorldArena Track 1 archive stores a 14-dimensional absolute joint
state in four ``joint_action`` datasets.  This module intentionally handles
only the action trajectory: the public archive has first-frame PNGs but no
per-timestep RGB datasets, so it is not by itself a video-training dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


ACTION_DIM = 14
ACTION_COMPONENT_KEYS = (
    "joint_action/left_arm",
    "joint_action/left_gripper",
    "joint_action/right_arm",
    "joint_action/right_gripper",
)
FLOWWAM_RGB_KEYS = (
    "observation/head_camera/rgb",
    "observation/left_camera/rgb",
    "observation/right_camera/rgb",
)


@dataclass(frozen=True)
class ActionNormStats:
    """Per-dimension z-score statistics compatible with FlowWAM."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,):
            raise ValueError(
                f"Action normalization must contain {ACTION_DIM} values, "
                f"got mean={mean.shape}, std={std.shape}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Action normalization contains NaN or Inf")
        if np.any(std < 0):
            raise ValueError("Action normalization standard deviations must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormStats":
        with np.load(path) as values:
            return cls(mean=values["mean"], std=values["std"])

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return ((actions - self.mean) / (self.std + 1.0e-6)).astype(np.float32, copy=False)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions * (self.std + 1.0e-6) + self.mean).astype(np.float32, copy=False)


def _as_column(values: np.ndarray, *, key: str, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape == (length,):
        return values[:, None]
    if values.shape == (length, 1):
        return values
    raise ValueError(f"{key} must have shape [T] or [T,1], got {values.shape}")


def read_action_sequence(
    path: str | Path,
    *,
    validate_vector: bool = True,
    vector_atol: float = 1.0e-5,
) -> np.ndarray:
    """Read ``[left arm, left gripper, right arm, right gripper]`` as [T,14]."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        missing = [key for key in ACTION_COMPONENT_KEYS if key not in handle]
        if missing:
            raise KeyError(f"{path}: missing action datasets: {missing}")

        left_arm = np.asarray(handle[ACTION_COMPONENT_KEYS[0]][:], dtype=np.float32)
        right_arm = np.asarray(handle[ACTION_COMPONENT_KEYS[2]][:], dtype=np.float32)
        if left_arm.ndim != 2 or left_arm.shape[1] != 6:
            raise ValueError(f"{path}: left_arm must be [T,6], got {left_arm.shape}")
        if right_arm.ndim != 2 or right_arm.shape[1] != 6:
            raise ValueError(f"{path}: right_arm must be [T,6], got {right_arm.shape}")
        length = int(left_arm.shape[0])
        if length < 1 or right_arm.shape[0] != length:
            raise ValueError(
                f"{path}: inconsistent/empty arm lengths: left={length}, right={right_arm.shape[0]}"
            )
        left_gripper = _as_column(handle[ACTION_COMPONENT_KEYS[1]][:], key="left_gripper", length=length)
        right_gripper = _as_column(handle[ACTION_COMPONENT_KEYS[3]][:], key="right_gripper", length=length)
        actions = np.concatenate(
            [left_arm, left_gripper, right_arm, right_gripper], axis=1
        ).astype(np.float32, copy=False)

        if validate_vector and "joint_action/vector" in handle:
            vector = np.asarray(handle["joint_action/vector"][:], dtype=np.float32)
            if vector.shape != actions.shape:
                raise ValueError(
                    f"{path}: joint_action/vector must be {actions.shape}, got {vector.shape}"
                )
            if not np.allclose(vector, actions, rtol=1.0e-5, atol=vector_atol):
                maximum = float(np.max(np.abs(vector - actions)))
                raise ValueError(
                    f"{path}: joint_action/vector ordering differs from the four components "
                    f"(max_abs_error={maximum:g})"
                )

    if actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
        raise ValueError(f"{path}: action sequence must be finite [T,{ACTION_DIM}], got {actions.shape}")
    return actions


def aligned_action_indices(
    *,
    action_frames: int,
    target_frames: int,
    video_frame_indices: Optional[Sequence[int]] = None,
    video_frame_count: Optional[int] = None,
    alignment_offset: int = 0,
) -> np.ndarray:
    """Map sampled video times to action times, preserving the first frame.

    If video indices are provided, their normalized timestamps are transferred
    to the HDF5 trajectory.  Thus unequal video/action lengths still align by
    relative time.  Positive offsets select a later action target.
    """

    if action_frames < 1 or target_frames < 1:
        raise ValueError("action_frames and target_frames must be positive")
    if video_frame_indices is None:
        positions = np.linspace(0, action_frames - 1, target_frames, dtype=np.float64)
    else:
        raw_video_indices = np.asarray(video_frame_indices)
        if raw_video_indices.shape != (target_frames,):
            raise ValueError(
                f"Expected {target_frames} video frame indices, got {raw_video_indices.shape}"
            )
        if video_frame_count is None or video_frame_count < 1:
            raise ValueError("video_frame_count is required with video_frame_indices")
        video_indices_f64 = raw_video_indices.astype(np.float64)
        if not np.isfinite(video_indices_f64).all():
            raise ValueError("video_frame_indices contains NaN or Inf")
        video_indices = np.rint(video_indices_f64).astype(np.int64)
        if not np.array_equal(video_indices_f64, video_indices.astype(np.float64)):
            raise ValueError("video_frame_indices must contain integer source-frame indices")
        if np.any(video_indices < 0) or np.any(video_indices >= video_frame_count):
            raise ValueError(
                f"video_frame_indices must be within [0,{video_frame_count - 1}], "
                f"got [{video_indices.min()},{video_indices.max()}]"
            )
        if np.any(np.diff(video_indices) < 0):
            raise ValueError("video_frame_indices must be monotonically non-decreasing")

        # This is the common paired-data case. Use the exact decoded video
        # indices with no floating-point timestamp conversion whatsoever.
        if action_frames == video_frame_count:
            positions = video_indices
        elif video_frame_count == 1:
            positions = np.zeros(target_frames, dtype=np.float64)
        else:
            positions = video_indices.astype(np.float64) / float(video_frame_count - 1) * float(action_frames - 1)

    indices = np.rint(positions).astype(np.int64)
    indices += int(alignment_offset)
    return np.clip(indices, 0, action_frames - 1)


def resample_action_sequence(
    actions: np.ndarray,
    *,
    target_frames: int,
    video_frame_indices: Optional[Sequence[int]] = None,
    video_frame_count: Optional[int] = None,
    alignment_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time-aligned [target_frames,14] actions and their source indices."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must be [T,{ACTION_DIM}], got {actions.shape}")
    indices = aligned_action_indices(
        action_frames=int(actions.shape[0]),
        target_frames=target_frames,
        video_frame_indices=video_frame_indices,
        video_frame_count=video_frame_count,
        alignment_offset=alignment_offset,
    )
    return actions[indices].astype(np.float32, copy=False), indices


def hdf5_leaf_keys(path: str | Path) -> set[str]:
    keys: set[str] = set()
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, value: keys.add(name) if isinstance(value, h5py.Dataset) else None)
    return keys


def flowwam_missing_rgb_keys(path: str | Path) -> list[str]:
    keys = hdf5_leaf_keys(path)
    return [key for key in FLOWWAM_RGB_KEYS if key not in keys]


def iter_worldarena_hdf5(root: str | Path) -> Iterable[Path]:
    """Yield manifest-referenced HDF5 files, or recursively discovered files."""

    root = Path(root)
    manifest = root / "questions.jsonl"
    if manifest.is_file():
        seen: set[Path] = set()
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            relative = record.get("trajectory_hdf5")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{manifest}:{line_number}: missing trajectory_hdf5")
            path = (root / relative).resolve()
            if path not in seen:
                seen.add(path)
                yield path
        return
    yield from sorted(root.rglob("*.hdf5"))


class WorldArenaActionIndex:
    """Resolve paired video samples to WorldArena-style HDF5 trajectories."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._aliases: dict[str, Optional[Path]] = {}
        paths = list(iter_worldarena_hdf5(self.root))
        if not paths:
            raise ValueError(f"No HDF5 trajectories found under {self.root}")
        for path in paths:
            self._register_path(path)
        self._register_questions_manifest()

    @staticmethod
    def _normal(value: str | Path) -> str:
        text = str(value).replace("\\", "/").strip("/")
        for suffix in (".hdf5", ".h5", ".mp4"):
            if text.lower().endswith(suffix):
                text = text[: -len(suffix)]
                break
        return text

    def _put(self, alias: str | Path, path: Path) -> None:
        key = self._normal(alias)
        if not key:
            return
        old = self._aliases.get(key)
        if old is None and key in self._aliases:
            return
        if old is not None and old != path:
            self._aliases[key] = None
        else:
            self._aliases[key] = path

    def _register_path(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(self.root)
        aliases = {
            relative,
            relative.with_suffix(""),
            path.name,
            path.stem,
            Path(relative.parent.name) / path.stem,
        }
        for alias in aliases:
            self._put(alias, path)

    def _register_questions_manifest(self) -> None:
        manifest = self.root / "questions.jsonl"
        if not manifest.is_file():
            return
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            relative = record["trajectory_hdf5"]
            path = (self.root / relative).resolve()
            aliases = [
                relative,
                record.get("question_id", ""),
                Path(str(record.get("question_id", ""))).name,
            ]
            task = record.get("task_name")
            episode = record.get("episode")
            if task is not None and episode is not None:
                aliases.append(f"{task}/episode{episode}")
            for alias in aliases:
                self._put(alias, path)

    def resolve(
        self,
        video_path: str | Path,
        *,
        dataset_root: Optional[str | Path] = None,
    ) -> Optional[Path]:
        video = Path(video_path)
        candidates: list[str | Path] = [video.name, video.stem, video.parent.name]
        if dataset_root is not None:
            try:
                relative = video.resolve().relative_to(Path(dataset_root).resolve())
            except ValueError:
                relative = None
            if relative is not None:
                candidates.extend(
                    [
                        relative,
                        relative.with_suffix(""),
                        relative.parent,
                        relative.parent.name,
                    ]
                )
        for candidate in candidates:
            value = self._aliases.get(self._normal(candidate))
            if value is not None:
                return value
        return None
