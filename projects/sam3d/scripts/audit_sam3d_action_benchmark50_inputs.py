#!/usr/bin/env python3
"""Strictly audit benchmark JSONL conditions before expensive inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import (
    read_action_sequence,
    resample_action_sequence,
)
from cosmos_predict2._src.predict2.inference.video2world import load_sam3d_condition_cache


EXPECTED = {
    "sam3d_tokens_B_F_N_D": (4, (1, 1, 1, 768)),
    "sam_mask_B_K_H_W": (4, (1, 8, 120, 160)),
    "sam_mask_meta_B_K_D": (3, (1, 8, 8)),
    "sam3d_geometry_B_C_H_W": (4, (1, 8, 120, 160)),
    "sam3d_shape_latents_B_K_N_D": (4, (1, 8, 256, 8)),
    "sam3d_object_pose_B_K_D": (3, (1, 8, 10)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--expected-fixed-frames", type=int)
    parser.add_argument("--expected-first-frame-resize-resolution")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rows, got {len(rows)}")

    seen: set[str] = set()
    native_lengths: list[int] = []
    zero_teacher = 0
    for row in rows:
        name = str(row["name"])
        if name in seen:
            raise ValueError(f"Duplicate sample name: {name}")
        seen.add(name)
        if args.expected_first_frame_resize_resolution is not None:
            actual_resize = row.get("first_frame_resize_resolution")
            if actual_resize != args.expected_first_frame_resize_resolution:
                raise ValueError(
                    f"{name}: expected first-frame pre-resize "
                    f"{args.expected_first_frame_resize_resolution}, got {actual_resize}"
                )
        for key in ("input_path", "sam3d_condition_path", "action_hdf5_path", "action_norm_path"):
            path = Path(row[key])
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {key}={path}")

        condition = load_sam3d_condition_cache(str(row["sam3d_condition_path"]))
        for key, (ndim, shape) in EXPECTED.items():
            tensor = condition[key]
            if tensor.ndim != ndim or tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: {key} expected {shape}, got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name}: {key} contains non-finite values")
        zero_teacher += int(torch.count_nonzero(condition["sam3d_tokens_B_F_N_D"]))

        raw_action = read_action_sequence(row["action_hdf5_path"], validate_vector=False)
        target_frames = int(row["num_output_frames"])
        aligned, indices = resample_action_sequence(raw_action, target_frames=target_frames)
        if aligned.shape != (target_frames, 14):
            raise ValueError(f"{name}: aligned action shape is {aligned.shape}")
        if indices[0] != 0 or indices[-1] != len(raw_action) - 1:
            raise ValueError(f"{name}: action endpoints not preserved: {indices[0]}, {indices[-1]}")
        if np.any(np.diff(indices) < 0) or not np.isfinite(aligned).all():
            raise ValueError(f"{name}: invalid/non-monotonic aligned action")
        if args.expected_fixed_frames is not None and target_frames != args.expected_fixed_frames:
            raise ValueError(f"{name}: expected {args.expected_fixed_frames} frames, got {target_frames}")
        with h5py.File(row["action_hdf5_path"], "r") as handle:
            native_lengths.append(int(handle["joint_action/left_arm"].shape[0]))

    print(
        "INPUT_AUDIT_OK "
        f"samples={len(rows)} condition_fields={len(EXPECTED)} action_dim=14 "
        f"native_frame_range=[{min(native_lengths)},{max(native_lengths)}] "
        f"teacher_token_nonzero_elements={zero_teacher}"
    )


if __name__ == "__main__":
    main()
