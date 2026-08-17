#!/usr/bin/env python3
"""Audit that a converted Cosmos checkpoint contains trained SAM/SAM3D paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


EXPECTED_FRAGMENTS = (
    "sam3d_token_projector",
    "sam_mask_tokenizer",
    "sam_mask_projector",
    "sam_mask_meta_projector",
    "sam_geometry_tokenizer",
    "sam_geometry_projector",
    "sam3d_shape_projector",
    "sam3d_pose_projector",
    "sam_modality_embeddings",
    "sam_context_block_gates",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(args.checkpoint)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint is not a non-empty state dictionary")
    tensor_state = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensor_state:
        raise TypeError("checkpoint contains no tensors")

    missing = [fragment for fragment in EXPECTED_FRAGMENTS if not any(fragment in key for key in tensor_state)]
    if missing:
        raise ValueError(f"missing SAM/SAM3D parameter groups: {missing}")
    for key, value in tensor_state.items():
        if any(fragment in key for fragment in EXPECTED_FRAGMENTS) and not torch.isfinite(value.float()).all():
            raise ValueError(f"non-finite SAM/SAM3D tensor: {key}")

    gate_items = [(key, value.float()) for key, value in tensor_state.items() if "sam_context_block_gates" in key]
    if len(gate_items) != 1:
        raise ValueError(f"expected one block-gate tensor, found {[key for key, _ in gate_items]}")
    gate_key, gates = gate_items[0]
    if gates.numel() != 28:
        raise ValueError(f"expected 28 DiT block gates, got {tuple(gates.shape)}")
    if torch.count_nonzero(gates) == 0:
        raise ValueError("all SAM context gates remain zero after training")

    matching = [key for key in tensor_state if any(fragment in key for fragment in EXPECTED_FRAGMENTS)]
    print(
        "SAM3D_CHECKPOINT_AUDIT_OK",
        args.checkpoint,
        f"state_tensors={len(tensor_state)}",
        f"sam_tensors={len(matching)}",
        f"gate_key={gate_key}",
        f"gate_abs_mean={float(gates.abs().mean()):.9g}",
        f"gate_abs_max={float(gates.abs().max()):.9g}",
    )


if __name__ == "__main__":
    main()
