# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic video dataset loader for Cosmos Predict2."""

import json
import os
import random
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import fcntl
import numpy as np
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
        included_batches: Optional[list[str]] = None,
        sampling_mode: str = "uniform",
        filter_short_videos: bool = True,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format
        self.included_batches = included_batches
        if sampling_mode not in {"continuous", "uniform"}:
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
        self.sampling_mode = sampling_mode
        self.caption_paths: Optional[list[str]] = None
        video_dir = os.path.join(self.dataset_dir, "videos")
        manifest_path = os.path.join(self.dataset_dir, "manifest.jsonl")

        if video_paths is None:
            if os.path.isfile(manifest_path):
                records = []
                with open(manifest_path, encoding="utf-8") as manifest_file:
                    for line_number, line in enumerate(manifest_file, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if "video" not in record or "instruction" not in record:
                            raise ValueError(f"Invalid manifest record at line {line_number}: {record}")
                        if self.included_batches is not None and record.get("batch") not in self.included_batches:
                            continue
                        records.append(record)
                self.video_paths = [os.path.join(self.dataset_dir, item["video"]) for item in records]
                self.caption_paths = [os.path.join(self.dataset_dir, item["instruction"]) for item in records]
                self.caption_format = "json"
            else:
                self._setup_caption_format()
                self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
                self.video_paths = sorted(self.video_paths)
        else:
            self._setup_caption_format()
            self.video_paths = video_paths

        if filter_short_videos:
            self._filter_videos_by_length()
        log.info(f"{len(self.video_paths)} videos in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def _filter_videos_by_length(self) -> None:
        """Remove videos shorter than the configured sequence length.

        Frame counts are cached in the dataset directory because every distributed
        worker constructs the dataset independently. The lock makes the initial
        scan happen only once on a shared filesystem.
        """
        cache_path = os.path.join(self.dataset_dir, ".video_frame_counts.json")
        lock_path = f"{cache_path}.lock"
        os.makedirs(self.dataset_dir, exist_ok=True)

        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with open(cache_path, encoding="utf-8") as cache_file:
                    cache = json.load(cache_file)
            except (FileNotFoundError, json.JSONDecodeError):
                cache = {}

            cache_changed = False
            frame_count_by_path: dict[str, int] = {}
            missing_entries: list[tuple[str, str]] = []
            trust_cache = os.environ.get("COSMOS_VIDEO_TRUST_FRAME_CACHE", "1") == "1"
            for video_path in self.video_paths:
                cached = cache.get(video_path)
                if trust_cache and cached is not None:
                    frame_count_by_path[video_path] = int(cached["frame_count"])
                    continue
                try:
                    stat = os.stat(video_path)
                    signature = f"{stat.st_size}:{stat.st_mtime_ns}"
                    if cached is not None and cached.get("signature") == signature:
                        frame_count_by_path[video_path] = int(cached["frame_count"])
                    else:
                        missing_entries.append((video_path, signature))
                except Exception as error:
                    log.warning(f"Failed to inspect video {video_path}; excluding it: {error}")
                    frame_count_by_path[video_path] = 0

            def inspect_video(entry: tuple[str, str]) -> tuple[str, str, int, Optional[str]]:
                video_path, signature = entry
                try:
                    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                    frame_count = len(reader)
                    del reader
                    return video_path, signature, frame_count, None
                except Exception as error:
                    return video_path, signature, 0, str(error)

            if missing_entries:
                scan_workers = min(int(os.environ.get("COSMOS_VIDEO_SCAN_WORKERS", "16")), len(missing_entries))
                log.info(
                    f"Inspecting {len(missing_entries)} uncached videos with {scan_workers} workers; "
                    f"{len(self.video_paths) - len(missing_entries)} frame counts were cached"
                )
                with ThreadPoolExecutor(max_workers=scan_workers) as executor:
                    for video_path, signature, frame_count, error in executor.map(inspect_video, missing_entries):
                        if error is not None:
                            log.warning(f"Failed to inspect video {video_path}; excluding it: {error}")
                        frame_count_by_path[video_path] = frame_count
                        cache[video_path] = {
                            "signature": signature,
                            "frame_count": frame_count,
                        }
                cache_changed = True

            frame_counts = [frame_count_by_path[video_path] for video_path in self.video_paths]

            if cache_changed:
                fd, temporary_path = tempfile.mkstemp(
                    prefix=".video_frame_counts.",
                    suffix=".tmp",
                    dir=self.dataset_dir,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as cache_file:
                        json.dump(cache, cache_file)
                    os.replace(temporary_path, cache_path)
                finally:
                    if os.path.exists(temporary_path):
                        os.unlink(temporary_path)

            keep_indices = [
                index for index, frame_count in enumerate(frame_counts) if frame_count >= self.sequence_length
            ]
            excluded_count = len(self.video_paths) - len(keep_indices)
            self.video_paths = [self.video_paths[index] for index in keep_indices]
            if self.caption_paths is not None:
                self.caption_paths = [self.caption_paths[index] for index in keep_indices]

            log.info(
                f"Kept {len(self.video_paths)} videos with at least {self.sequence_length} frames; "
                f"excluded {excluded_count} shorter or unreadable videos"
            )

            if not self.video_paths:
                raise ValueError(
                    f"No readable videos with at least {self.sequence_length} frames were found in {self.dataset_dir}"
                )

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _sample_frame_ids(self, total_frames: int) -> np.ndarray:
        """Choose source-frame indices for one sample.

        Keeping this operation separate lets sidecar modalities (for example
        robot actions) use the exact same timestamps as the decoded video.
        """

        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video has only {total_frames} frames, at least "
                f"{self.sequence_length} frames are required."
            )
        if self.sampling_mode == "continuous":
            max_start_idx = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start_idx + 1)
            return np.arange(start_frame, start_frame + self.sequence_length, dtype=np.int64)
        # Uniform sampling spans the complete timeline, preserves frame zero,
        # and includes the last source frame.
        return np.rint(np.linspace(0, total_frames - 1, self.sequence_length)).astype(np.int64)

    def _load_video_with_frame_ids(
        self, video_path: str
    ) -> tuple[np.ndarray, float, np.ndarray, int]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        frame_ids = self._sample_frame_ids(total_frames)

        frame_data = vr.get_batch(frame_ids.tolist()).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        # Uniform sampling spans the full source duration, so report its
        # effective FPS rather than the source container FPS.
        if self.sampling_mode == "uniform" and total_frames > 1:
            fps = fps * (self.sequence_length - 1) / (total_frames - 1)
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, frame_ids, total_frames

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        frame_data, fps, _, _ = self._load_video_with_frame_ids(video_path)
        return frame_data, fps

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            if isinstance(data.get("instruction"), str):
                return data["instruction"].strip()

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    log.warning(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            log.warning(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_frames_with_frame_ids(
        self, video_path: str
    ) -> tuple[torch.Tensor, float, np.ndarray, int]:
        frames, fps, frame_ids, total_frames = self._load_video_with_frame_ids(video_path)
        return self._preprocess_frames(frames), fps, frame_ids, total_frames

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps, _, _ = self._get_frames_with_frame_ids(video_path)
        return frames, fps

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = (
                    self.caption_paths[index]
                    if self.caption_paths is not None
                    else os.path.join(self.caption_dir, f"{video_basename}.json")
                )
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["ai_caption"] = caption

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,  # Ignore extra arguments
) -> DataLoader:
    """Create DataLoader with commonly used parameters.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        sampler: Optional sampler for data loading
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA transfer
        drop_last: Drop incomplete last batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        **kwargs: Extra arguments (ignored)

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_train_val_dataloaders(
    dataset_path: str, val_percentage: float, seed: int, video_size: tuple[int, int] = (704, 1280)
):
    video_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(video_dir):
        log.debug(f"Dataset path {dataset_path} does not exist, returning empty dataloaders")
        return dict(), dict()
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    random.seed(seed)
    random.shuffle(video_paths)

    cutoff = int(len(video_paths) * val_percentage)
    val_video_paths = video_paths[:cutoff]
    train_video_paths = video_paths[cutoff:]

    def get_dataset(video_paths):
        return L(VideoDataset)(
            video_paths=video_paths,
            num_frames=121,
            video_size=video_size,
            dataset_dir=dataset_path,
        )

    ipn_hand_train_dataset = get_dataset(train_video_paths)
    ipn_hand_val_dataset = get_dataset(val_video_paths)

    def get_dataloader(dataset):
        return L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    return get_dataloader(ipn_hand_train_dataset), get_dataloader(ipn_hand_val_dataset)
