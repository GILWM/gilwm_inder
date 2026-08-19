#!/usr/bin/env python3
"""Audit normalized WorldArena action ranges for numerical outliers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import (
    ActionNormStats,
    read_action_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("norm", type=Path)
    parser.add_argument("--min-video-frames", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()

    root = args.manifest.parent
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    records = [record for record in records if int(record.get("frame_count", 0)) >= args.min_video_frames]
    stats = ActionNormStats.load(args.norm)
    thresholds = (5.0, 10.0, 20.0, 50.0, 100.0, 500.0)
    maximum = np.zeros_like(stats.mean, dtype=np.float64)
    maximum_path = [""] * len(maximum)
    counts = np.zeros(len(thresholds), dtype=np.int64)
    total_values = 0

    for index, record in enumerate(records, 1):
        relative = record.get("actions") or record.get("trajectory_hdf5") or record.get("action_hdf5")
        if not relative:
            raise ValueError(f"Record {index} has no action HDF5 path")
        path = (root / relative).resolve()
        normalized = np.abs(stats.normalize(read_action_sequence(path, validate_vector=False))).astype(np.float64)
        current = normalized.max(axis=0)
        for dim in np.flatnonzero(current > maximum):
            maximum_path[int(dim)] = str(path)
        maximum = np.maximum(maximum, current)
        counts += np.asarray([(normalized > threshold).sum() for threshold in thresholds], dtype=np.int64)
        total_values += normalized.size
        if args.progress_every and index % args.progress_every == 0:
            print(f"AUDITED={index}/{len(records)} MAX_ABS_Z={maximum.max():.6g}", flush=True)

    report = {
        "manifest": str(args.manifest.resolve()),
        "norm": str(args.norm.resolve()),
        "min_video_frames": args.min_video_frames,
        "records": len(records),
        "total_values": total_values,
        "max_abs_z_by_dimension": maximum.tolist(),
        "max_abs_z": float(maximum.max()),
        "max_paths_by_dimension": maximum_path,
        "threshold_counts": {str(threshold): int(count) for threshold, count in zip(thresholds, counts)},
    }
    print(json.dumps(report, indent=2))
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
