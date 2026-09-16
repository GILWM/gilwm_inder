#!/usr/bin/env python3
"""Audit real manifest samples for strict video/action timestamp alignment."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from decord import VideoReader, cpu

from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import (
    ACTION_COMPONENT_KEYS,
    hdf5_leaf_keys,
    read_action_sequence,
    resample_action_sequence,
)


ACTION_RECORD_KEYS = ("trajectory_hdf5", "action_hdf5", "hdf5", "actions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--max-samples", type=int, default=0, help="0 audits every record")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--decode-video",
        action="store_true",
        help="Read actual video lengths instead of trusting manifest frame_count",
    )
    parser.add_argument("--json-report", type=Path)
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_records(manifests: list[Path]) -> list[tuple[Path, int, dict[str, Any]]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                records.append((manifest, line_number, json.loads(line)))
    return records


def main() -> None:
    args = parse_args()
    all_records = load_records(args.manifest)
    manifest_short_records = []
    records = []
    for item in all_records:
        frame_count = item[2].get("frame_count")
        if frame_count is not None and int(frame_count) < args.num_frames:
            manifest_short_records.append(item)
        else:
            records.append(item)
    if args.max_samples > 0 and len(records) > args.max_samples:
        records = random.Random(args.seed).sample(records, args.max_samples)

    summary: Counter[str] = Counter()
    failures: list[str] = []
    extra_key_counts: Counter[str] = Counter()

    for manifest, line_number, record in records:
        label = f"{manifest}:{line_number}"
        try:
            video_value = record.get("video")
            action_value = next((record.get(key) for key in ACTION_RECORD_KEYS if record.get(key)), None)
            if not isinstance(video_value, str) or not isinstance(action_value, str):
                raise ValueError("missing string video/action field")
            video_path = resolve(manifest.parent, video_value)
            action_path = resolve(manifest.parent, action_value)
            if not video_path.is_file() or not action_path.is_file():
                raise FileNotFoundError(f"video={video_path.is_file()} action={action_path.is_file()}")

            manifest_video_frames = (
                int(record["frame_count"]) if record.get("frame_count") is not None else None
            )
            if args.decode_video:
                reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
                video_frames = len(reader)
                del reader
                if manifest_video_frames is not None and manifest_video_frames != video_frames:
                    raise ValueError(
                        f"manifest frame_count={manifest_video_frames}, decoded video length={video_frames}"
                    )
            else:
                if manifest_video_frames is None:
                    raise ValueError("frame_count is required unless --decode-video is used")
                video_frames = manifest_video_frames
            if video_frames < args.num_frames:
                raise ValueError(f"video has {video_frames} frames, needs {args.num_frames}")

            # Only the four official component datasets are read. All other
            # leaf datasets are diagnostic metadata and cannot affect values.
            actions = read_action_sequence(action_path, validate_vector=False)
            leaf_keys = hdf5_leaf_keys(action_path)
            extra_keys = leaf_keys.difference(ACTION_COMPONENT_KEYS)
            for key in extra_keys:
                extra_key_counts[key] += 1
            if extra_keys:
                summary["h5_with_extra_datasets"] += 1

            uniform_video_indices = np.rint(
                np.linspace(0, video_frames - 1, args.num_frames)
            ).astype(np.int64)
            sampled, action_indices = resample_action_sequence(
                actions,
                target_frames=args.num_frames,
                video_frame_indices=uniform_video_indices,
                video_frame_count=video_frames,
            )
            np.testing.assert_array_equal(sampled, actions[action_indices])

            if len(actions) == video_frames:
                np.testing.assert_array_equal(action_indices, uniform_video_indices)
                # Also cover every possible continuous-crop boundary class.
                max_start = video_frames - args.num_frames
                for start in sorted({0, max_start // 2, max_start}):
                    continuous_video_indices = np.arange(start, start + args.num_frames, dtype=np.int64)
                    _, continuous_action_indices = resample_action_sequence(
                        actions,
                        target_frames=args.num_frames,
                        video_frame_indices=continuous_video_indices,
                        video_frame_count=video_frames,
                    )
                    np.testing.assert_array_equal(continuous_action_indices, continuous_video_indices)
                summary["equal_length_exact_index_samples"] += 1
            else:
                summary["relative_timestamp_samples"] += 1

            summary["valid_samples"] += 1
            summary["video_frames"] += video_frames
            summary["action_frames"] += len(actions)
        except Exception as error:
            failures.append(f"{label}: {error}")

    report = {
        "manifests": [str(path) for path in args.manifest],
        "manifest_records": len(all_records),
        "manifest_short_records_excluded_like_training": len(manifest_short_records),
        "requested_samples": len(records),
        "num_frames": args.num_frames,
        "decoded_actual_video_lengths": bool(args.decode_video),
        **dict(summary),
        "extra_dataset_counts": dict(extra_key_counts.most_common()),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"video/action alignment audit failed for {len(failures)} samples")


if __name__ == "__main__":
    main()
