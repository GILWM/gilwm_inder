#!/usr/bin/env python3
"""Prepare deterministic Cosmos AR inference shards for WorldArena benchmark50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--name-prefix", default="iter1000_ar93_640x480")
    parser.add_argument("--num-steps", type=int, default=35)
    args = parser.parse_args()

    questions = args.dataset_root / "dataset_track1" / "questions.jsonl"
    rows = [
        json.loads(line)
        for line in questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 50:
        raise ValueError(f"Expected 50 benchmark rows, found {len(rows)} in {questions}")

    samples: list[dict[str, object]] = []
    for row in rows:
        episode = int(row["episode"])
        first_frame = args.dataset_root / "dataset_track1" / row["first_frame"]
        if not first_frame.is_file():
            raise FileNotFoundError(first_frame)
        samples.append(
            {
                "inference_type": "image2world",
                "name": f"{args.name_prefix}_episode{episode:04d}",
                "input_path": str(first_frame),
                "prompt": row["instruction"],
                "resolution": "480,640",
                "num_output_frames": 121,
                "num_steps": args.num_steps,
                "seed": episode,
                "guidance": 7,
                "enable_autoregressive": True,
                "chunk_size": 93,
                "chunk_overlap": 1,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "inputs_640x480_121f_ar93_all.jsonl"
    all_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    for shard in range(args.shards):
        selected = samples[shard :: args.shards]
        shard_path = args.output_dir / f"shard_{shard:02d}.jsonl"
        shard_path.write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in selected),
            encoding="utf-8",
        )
        print(f"{shard_path}: {len(selected)}")


if __name__ == "__main__":
    main()
