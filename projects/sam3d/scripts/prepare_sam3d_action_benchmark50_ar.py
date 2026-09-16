#!/usr/bin/env python3
"""Prepare benchmark50 inputs with explicit SAM/SAM3D and HDF5 action conditions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def episode_number(path: Path) -> int:
    return int(path.stem.removeprefix("episode"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--action-norm-path", type=Path, required=True)
    parser.add_argument("--sam3d-condition-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=int, default=7)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--frame-count-mode",
        choices=("fixed", "action"),
        default="fixed",
        help="Use a fixed output length or exactly match each HDF5 action trajectory.",
    )
    parser.add_argument("--num-output-frames", type=int, default=93)
    parser.add_argument("--chunk-size", type=int, default=93)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument(
        "--first-frame-resize-resolution",
        help="Optional H,W pre-resize for the image first frame; omitted keeps the original path unchanged.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.shards < 1:
        raise ValueError(f"--shards must be positive, got {args.shards}")
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        raise ValueError("chunk overlap must satisfy 0 <= overlap < chunk size")
    if not args.action_norm_path.is_file():
        raise FileNotFoundError(args.action_norm_path)
    if not args.sam3d_condition_root.is_dir():
        raise FileNotFoundError(args.sam3d_condition_root)

    first_frames = sorted(
        (args.benchmark_root / "first_frame").glob("episode*.png"),
        key=episode_number,
    )
    if len(first_frames) != 50:
        raise ValueError(f"Expected 50 first frames, found {len(first_frames)}")

    samples: list[dict[str, object]] = []
    frame_plan: list[dict[str, int | str]] = []
    for first_frame in first_frames:
        episode = episode_number(first_frame)
        instruction_path = args.benchmark_root / "instruction" / f"episode{episode}.json"
        action_path = args.benchmark_root / "action_hdf5" / f"episode{episode}.hdf5"
        condition_path = args.sam3d_condition_root / f"episode_{episode:06d}.pt"
        for required in (instruction_path, action_path, condition_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        with action_path.open("rb") as stream:
            if stream.read(len(HDF5_MAGIC)) != HDF5_MAGIC:
                raise ValueError(f"{action_path} is not materialized HDF5 data")

        import h5py

        with h5py.File(action_path, "r") as handle:
            action_frames = int(handle["joint_action/left_arm"].shape[0])
        output_frames = action_frames if args.frame_count_mode == "action" else args.num_output_frames
        if output_frames < 1:
            raise ValueError(f"Invalid output frame count for episode {episode}: {output_frames}")
        effective_chunk = args.chunk_size - args.chunk_overlap
        chunks = 1 + max(0, math.ceil((output_frames - args.chunk_size) / effective_chunk))
        frame_plan.append(
            {
                "episode": episode,
                "action_frames": action_frames,
                "output_frames": output_frames,
                "chunks": chunks,
                "condition": str(condition_path),
            }
        )
        instruction = json.loads(instruction_path.read_text(encoding="utf-8"))["instruction"]
        sample: dict[str, object] = {
                "inference_type": "image2world",
                "name": f"{args.name_prefix}_episode{episode:04d}",
                "input_path": str(first_frame),
                "prompt": instruction,
                "sam3d_condition_path": str(condition_path),
                "action_hdf5_path": str(action_path),
                "action_norm_path": str(args.action_norm_path),
                "resolution": "480,640",
                "fps": args.fps,
                "num_output_frames": output_frames,
                "num_steps": args.num_steps,
                "seed": args.seed_offset + episode,
                "guidance": args.guidance,
                "enable_autoregressive": True,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
        }
        if args.first_frame_resize_resolution is not None:
            sample["first_frame_resize_resolution"] = args.first_frame_resize_resolution
        samples.append(sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (args.output_dir / "smoke.json").write_text(
        json.dumps(samples[0], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "frame_plan.json").write_text(
        json.dumps(frame_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.frame_count_mode == "action":
        shard_samples: list[list[dict[str, object]]] = [[] for _ in range(args.shards)]
        shard_costs = [0] * args.shards
        ordered = sorted(
            zip(samples, frame_plan, strict=True),
            key=lambda item: (int(item[1]["chunks"]), int(item[1]["output_frames"])),
            reverse=True,
        )
        for sample, plan in ordered:
            target = min(range(args.shards), key=lambda shard: (shard_costs[shard], shard))
            shard_samples[target].append(sample)
            shard_costs[target] += int(plan["chunks"])
    else:
        shard_samples = [samples[shard :: args.shards] for shard in range(args.shards)]
        shard_costs = [
            sum(int(frame_plan[index]["chunks"]) for index in range(shard, len(samples), args.shards))
            for shard in range(args.shards)
        ]

    for shard, selected in enumerate(shard_samples):
        shard_path = args.output_dir / f"shard_{shard:02d}.jsonl"
        shard_path.write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in selected),
            encoding="utf-8",
        )
        print(f"{shard_path}: {len(selected)} samples, {shard_costs[shard]} AR chunks")
    print(
        "FRAME_PLAN "
        f"mode={args.frame_count_mode} samples={len(samples)} "
        f"frames={sum(int(item['output_frames']) for item in frame_plan)} "
        f"range=[{min(int(item['output_frames']) for item in frame_plan)},"
        f"{max(int(item['output_frames']) for item in frame_plan)}] "
        f"chunks={sum(int(item['chunks']) for item in frame_plan)} "
        f"shard_chunks={shard_costs}"
    )


if __name__ == "__main__":
    main()
