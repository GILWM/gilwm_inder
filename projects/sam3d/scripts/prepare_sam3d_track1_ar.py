#!/usr/bin/env python3
"""Prepare the official WorldArena2 Track1 set for SAM3D AR inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "questions.jsonl").is_file():
        return path
    nested = path / "dataset_track1"
    if (nested / "questions.jsonl").is_file():
        return nested
    raise FileNotFoundError(f"questions.jsonl not found under {path}")


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG first frame: {path}")
    return struct.unpack(">II", header[16:24])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=os.environ.get("WA2_TRACK1_DATASET_ROOT"),
        help="Track1 directory itself, or its parent; defaults to WA2_TRACK1_DATASET_ROOT",
    )
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument(
        "--seed-salt",
        default="",
        help="If set, derive an independent deterministic seed per episode from this salt",
    )
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("dataset_root or WA2_TRACK1_DATASET_ROOT is required")
    if args.shards <= 0:
        parser.error("--shards must be positive")

    dataset = resolve_dataset_root(Path(args.dataset_root))
    questions = dataset / "questions.jsonl"
    rows = [
        json.loads(line)
        for line in questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 Track1 questions, got {len(rows)}")
    episodes = [int(row["episode"]) for row in rows]
    if episodes != list(range(1, 1001)):
        raise ValueError("Track1 episode IDs must be exactly 1..1000 in order")

    resolutions: dict[str, int] = {}
    samples: list[dict[str, object]] = []
    for row, episode in zip(rows, episodes):
        first_frame = (dataset / row["first_frame"]).resolve()
        if not first_frame.is_file():
            raise FileNotFoundError(first_frame)
        width, height = png_size(first_frame)
        key = f"{width}x{height}"
        resolutions[key] = resolutions.get(key, 0) + 1
        if args.seed_salt:
            digest = hashlib.blake2s(
                f"{args.seed_salt}:{episode}".encode("utf-8"), digest_size=8
            ).digest()
            seed = int.from_bytes(digest, "big") % (2**31 - 1)
        else:
            seed = episode
        samples.append(
            {
                "inference_type": "image2world",
                "name": f"episode_{episode:06d}",
                "input_path": str(first_frame),
                "prompt": row["instruction"].strip(),
                "resolution": "480,640",
                "num_output_frames": 121,
                "num_steps": args.num_steps,
                "seed": seed,
                "guidance": args.guidance,
                "enable_autoregressive": True,
                "chunk_size": 93,
                "chunk_overlap": 1,
            }
        )

    inputs = args.eval_root / "inputs"
    output = args.eval_root / "videos"
    logs = args.eval_root / "logs"
    status = args.eval_root / "status"
    for directory in (inputs, output, logs, status):
        directory.mkdir(parents=True, exist_ok=True)

    all_path = inputs / "all.jsonl"
    all_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    shard_sizes = []
    for shard in range(args.shards):
        selected = samples[shard :: args.shards]
        shard_sizes.append(len(selected))
        (inputs / f"shard_{shard:02d}.jsonl").write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in selected),
            encoding="utf-8",
        )

    manifest = {
        "dataset_root": str(dataset),
        "questions": str(questions),
        "eval_root": str(args.eval_root.resolve()),
        "output": str(output.resolve()),
        "samples": len(samples),
        "shards": args.shards,
        "shard_sizes": shard_sizes,
        "source_resolutions": resolutions,
        "output_resolution": "640x480",
        "output_frames": 121,
        "output_fps": 24,
        "autoregressive": True,
        "chunk_size": 93,
        "chunk_overlap": 1,
        "guidance": args.guidance,
        "seed_salt": args.seed_salt or None,
        "sam3d_condition_sidecars": False,
        "input_policy": "official prompt and first frame only; no GT/future-frame leakage",
    }
    (args.eval_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
