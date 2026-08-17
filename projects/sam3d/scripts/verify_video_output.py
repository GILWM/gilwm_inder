#!/usr/bin/env python3
"""Validate a generated video inside the independent Cosmos runtime."""

import argparse
from pathlib import Path

from decord import VideoReader, cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()

    if not args.video.is_file() or args.video.stat().st_size <= 0:
        raise FileNotFoundError(args.video)
    reader = VideoReader(str(args.video), ctx=cpu(0), num_threads=2)
    frame_count = len(reader)
    first = reader[0]
    height, width = map(int, first.shape[:2])
    if (width, height, frame_count) != (args.width, args.height, args.frames):
        raise ValueError(
            f"Unexpected video: width={width}, height={height}, frames={frame_count}; "
            f"expected {args.width}x{args.height}, {args.frames} frames"
        )
    print(
        "VIDEO_OUTPUT_VALID",
        args.video,
        f"width={width}",
        f"height={height}",
        f"frames={frame_count}",
    )


if __name__ == "__main__":
    main()
