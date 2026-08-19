#!/usr/bin/env python3
"""Audit WorldArena-style HDF5 action trajectories and write z-score stats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import (
    ACTION_DIM,
    FLOWWAM_RGB_KEYS,
    ActionNormStats,
    hdf5_leaf_keys,
    iter_worldarena_hdf5,
    read_action_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="WorldArena dataset_track1 or another HDF5 root")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--write-norm", type=Path)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--skip-vector-validation", action="store_true")
    args = parser.parse_args()

    paths = list(iter_worldarena_hdf5(args.root))
    failures: list[str] = []
    lengths: list[int] = []
    schema_counts: Counter[tuple[str, ...]] = Counter()
    flowwam_rgb_complete = 0
    total_steps = 0
    action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    action_sum_sq = np.zeros(ACTION_DIM, dtype=np.float64)

    for path in paths:
        try:
            actions = read_action_sequence(
                path, validate_vector=not args.skip_vector_validation
            )
            keys = hdf5_leaf_keys(path)
            schema_counts[tuple(sorted(keys))] += 1
            flowwam_rgb_complete += int(all(key in keys for key in FLOWWAM_RGB_KEYS))
            lengths.append(int(actions.shape[0]))
            total_steps += int(actions.shape[0])
            action_sum += actions.sum(axis=0, dtype=np.float64)
            action_sum_sq += np.square(actions, dtype=np.float64).sum(axis=0)
        except Exception as error:  # report all bad episodes in one audit
            failures.append(f"{path}: {type(error).__name__}: {error}")

    if total_steps:
        mean = action_sum / total_steps
        variance = np.maximum(action_sum_sq / total_steps - np.square(mean), 0.0)
        std = np.sqrt(variance)
    else:
        mean = np.zeros(ACTION_DIM, dtype=np.float64)
        std = np.zeros(ACTION_DIM, dtype=np.float64)

    report = {
        "root": str(args.root.resolve()),
        "hdf5_files": len(paths),
        "valid_action_files": len(paths) - len(failures),
        "invalid_action_files": len(failures),
        "total_action_steps": total_steps,
        "min_action_frames": min(lengths) if lengths else None,
        "max_action_frames": max(lengths) if lengths else None,
        "action_dim": ACTION_DIM,
        "unique_leaf_schemas": len(schema_counts),
        "flowwam_rgb_complete_files": flowwam_rgb_complete,
        "flowwam_direct_video_training_compatible": bool(paths) and flowwam_rgb_complete == len(paths),
        "action_mean": mean.astype(np.float32).tolist(),
        "action_std": std.astype(np.float32).tolist(),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.write_norm is not None and not failures and total_steps:
        args.write_norm.parent.mkdir(parents=True, exist_ok=True)
        ActionNormStats(mean=mean, std=std).save(args.write_norm)
        print(f"ACTION_NORM_WRITTEN={args.write_norm.resolve()}")
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if args.expected_count is not None and len(paths) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} HDF5 files, found {len(paths)}"
        )
    if failures:
        raise SystemExit(f"{len(failures)} invalid HDF5 action files")


if __name__ == "__main__":
    main()
