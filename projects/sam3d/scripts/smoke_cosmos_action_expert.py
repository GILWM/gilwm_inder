#!/usr/bin/env python3
"""Run an isolated realistic-shape forward/backward for CosmosActionExpertHead."""

from __future__ import annotations

import argparse
import json

import torch

from cosmos_predict2._src.predict2.sam3d.action_alignment import CosmosActionExpertHead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-frames", type=int, default=24)
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=40)
    parser.add_argument("--video-layers", type=int, default=6)
    parser.add_argument("--action-frames", type=int, default=93)
    args = parser.parse_args()

    device = torch.device(args.device)
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "reset_peak_memory_stats"):
        backend.reset_peak_memory_stats(device)

    head = CosmosActionExpertHead(model_dim=2048).to(device)
    token_count = args.latent_frames * args.grid_height * args.grid_width
    features = [
        torch.randn(
            args.batch_size,
            token_count,
            2048,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(args.video_layers)
    ]
    actions = torch.randn(args.batch_size, args.action_frames, 14, device=device)
    timesteps = torch.full((args.batch_size, 1), 0.5, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
        losses = head(
            features,
            actions,
            latent_frames=args.latent_frames,
            spatial_shape=(args.grid_height, args.grid_width),
            timesteps_B_T=timesteps,
        )
        total = losses.prediction + 0.1 * losses.alignment
    total.backward()
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()
    peak_bytes = (
        int(backend.max_memory_allocated(device))
        if backend is not None and hasattr(backend, "max_memory_allocated")
        else None
    )
    print(
        json.dumps(
            {
                "parameters": head.param_count(),
                "prediction_loss": float(losses.prediction.detach().cpu()),
                "alignment_loss": float(losses.alignment.detach().cpu()),
                "peak_allocated_gib": None if peak_bytes is None else peak_bytes / 1024**3,
                "feature_shape": [args.batch_size, token_count, 2048],
                "feature_layers": args.video_layers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
