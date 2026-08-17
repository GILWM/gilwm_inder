#!/usr/bin/env python3
"""Safely validate the locally archived SAM 3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-size", type=int, default=3_450_062_241)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_size = args.checkpoint.stat().st_size
    if actual_size != args.expected_size:
        raise RuntimeError(
            f"SAM3 size mismatch: expected={args.expected_size} actual={actual_size}"
        )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"Expected a non-empty mapping, got {type(payload)!r}")

    state_dict = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError("SAM3 checkpoint does not contain a non-empty state dict")

    tensor_items = [(key, value) for key, value in state_dict.items() if torch.is_tensor(value)]
    if not tensor_items:
        raise TypeError("SAM3 checkpoint state dict contains no tensors")

    parameter_count = sum(value.numel() for _, value in tensor_items)
    first_keys = sorted(key for key, _ in tensor_items)[:8]
    print(
        "SAM3_CHECKPOINT_OK",
        f"bytes={actual_size}",
        f"top_level_keys={sorted(str(key) for key in payload)[:8]}",
        f"tensor_keys={len(tensor_items)}",
        f"parameters={parameter_count}",
        f"first_tensor_keys={first_keys}",
    )


if __name__ == "__main__":
    main()
