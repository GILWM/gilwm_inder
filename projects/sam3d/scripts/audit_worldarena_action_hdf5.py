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
    parser.add_argument(
        "roots",
        type=Path,
        nargs="+",
        help="One or more WorldArena/filtered-manifest HDF5 roots",
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--min-video-frames",
        type=int,
        help="For compact manifest roots, include only records meeting this video length",
    )
    parser.add_argument("--write-norm", type=Path)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--skip-vector-validation", action="store_true")
    parser.add_argument(
        "--skip-schema-audit",
        action="store_true",
        help="Skip the second HDF5 open used only for RGB/schema reporting",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    def selected_paths(root: Path):
        manifest = root / "manifest.jsonl"
        if args.min_video_frames is None or not manifest.is_file():
            yield from iter_worldarena_hdf5(root)
            return
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("frame_count", 0)) < args.min_video_frames:
                continue
            relative = next(
                (record.get(key) for key in ("trajectory_hdf5", "action_hdf5", "hdf5", "actions") if record.get(key)),
                None,
            )
            if not isinstance(relative, str):
                raise ValueError(f"{manifest}:{line_number}: missing action HDF5 path")
            yield root / relative

    paths_by_resolved_path = {path.absolute(): path for root in args.roots for path in selected_paths(root)}
    paths = [paths_by_resolved_path[key] for key in sorted(paths_by_resolved_path, key=str)]
    failures: list[str] = []
    lengths: list[int] = []
    schema_counts: Counter[tuple[str, ...]] = Counter()
    flowwam_rgb_complete = 0
    total_steps = 0
    action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    action_sum_sq = np.zeros(ACTION_DIM, dtype=np.float64)

    for index, path in enumerate(paths, 1):
        try:
            actions = read_action_sequence(path, validate_vector=not args.skip_vector_validation)
            if not args.skip_schema_audit:
                keys = hdf5_leaf_keys(path)
                schema_counts[tuple(sorted(keys))] += 1
                flowwam_rgb_complete += int(all(key in keys for key in FLOWWAM_RGB_KEYS))
            lengths.append(int(actions.shape[0]))
            total_steps += int(actions.shape[0])
            action_sum += actions.sum(axis=0, dtype=np.float64)
            action_sum_sq += np.square(actions, dtype=np.float64).sum(axis=0)
        except Exception as error:  # report all bad episodes in one audit
            failures.append(f"{path}: {type(error).__name__}: {error}")
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(
                f"AUDIT_PROGRESS={index}/{len(paths)} failures={len(failures)} steps={total_steps}",
                flush=True,
            )

    if total_steps:
        mean = action_sum / total_steps
        variance = np.maximum(action_sum_sq / total_steps - np.square(mean), 0.0)
        std = np.sqrt(variance)
    else:
        mean = np.zeros(ACTION_DIM, dtype=np.float64)
        std = np.zeros(ACTION_DIM, dtype=np.float64)

    report = {
        "roots": [str(root.resolve()) for root in args.roots],
        "min_video_frames": args.min_video_frames,
        "hdf5_files": len(paths),
        "valid_action_files": len(paths) - len(failures),
        "invalid_action_files": len(failures),
        "total_action_steps": total_steps,
        "min_action_frames": min(lengths) if lengths else None,
        "max_action_frames": max(lengths) if lengths else None,
        "action_dim": ACTION_DIM,
        "schema_audit_skipped": args.skip_schema_audit,
        "unique_leaf_schemas": None if args.skip_schema_audit else len(schema_counts),
        "flowwam_rgb_complete_files": None if args.skip_schema_audit else flowwam_rgb_complete,
        "flowwam_direct_video_training_compatible": (
            None if args.skip_schema_audit else bool(paths) and flowwam_rgb_complete == len(paths)
        ),
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
        args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.expected_count is not None and len(paths) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} HDF5 files, found {len(paths)}")
    if failures:
        raise SystemExit(f"{len(failures)} invalid HDF5 action files")


if __name__ == "__main__":
    main()
