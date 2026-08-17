#!/usr/bin/env python3
"""Audit completeness and video geometry for the 50-sample SAM3D evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("videos", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.inputs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        path = args.videos / f"{row['name']}.mp4"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path.name)
            continue
        video = cv2.VideoCapture(str(path))
        if not video.isOpened():
            invalid.append(f"{path.name}: cannot open")
            continue
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        video.release()
        if (width, height, frames) != (640, 480, 121) or abs(fps - 24.0) > 0.01:
            invalid.append(
                f"{path.name}: width={width} height={height} frames={frames} fps={fps:g}"
            )

    print(f"EXPECTED={len(rows)}")
    print(f"VALID={len(rows) - len(missing) - len(invalid)}")
    print(f"MISSING={len(missing)}")
    print(f"INVALID={len(invalid)}")
    for item in missing:
        print(f"MISSING_FILE {item}")
    for item in invalid:
        print(f"INVALID_FILE {item}")
    if missing or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
