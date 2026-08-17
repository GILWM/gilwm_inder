#!/usr/bin/env python3
"""Compare deterministic Cosmos/SAM3D checkpoint videos against one source clip.

The metrics are intended for controlled checkpoint comparisons, not as a
replacement for a multi-prompt human or benchmark evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def read_video(path: Path, count: int | None = None) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=4)
    if count is None:
        indices = np.arange(len(reader), dtype=np.int64)
    else:
        indices = np.rint(np.linspace(0, len(reader) - 1, count)).astype(np.int64)
        indices[0], indices[-1] = 0, len(reader) - 1
    frames = reader.get_batch(indices.tolist()).asnumpy()
    del reader
    return frames


def pixel_metrics(frames: np.ndarray) -> dict[str, float]:
    value = torch.from_numpy(frames).float() / 255.0
    velocity = value[1:] - value[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    gray = value.mean(dim=-1)
    laplacian = (
        -4 * gray[:, 1:-1, 1:-1]
        + gray[:, :-2, 1:-1]
        + gray[:, 2:, 1:-1]
        + gray[:, 1:-1, :-2]
        + gray[:, 1:-1, 2:]
    )
    return {
        "frame_delta_l1": velocity.abs().mean().item(),
        "frame_accel_l1": acceleration.abs().mean().item(),
        "sharpness_laplacian_var": laplacian.var(dim=(1, 2)).mean().item(),
    }


class DinoEncoder:
    def __init__(self, repo: Path, device: str):
        self.device = torch.device(device)
        self.model = torch.hub.load(
            str(repo), "dinov2_vitb14", source="local", pretrained=True, verbose=False
        ).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, frames: np.ndarray, batch_size: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(frames), batch_size):
            value = torch.from_numpy(frames[start : start + batch_size]).permute(0, 3, 1, 2)
            value = value.to(self.device, dtype=torch.float32).div_(255.0)
            value = F.interpolate(value, (224, 224), mode="bilinear", align_corners=False)
            value = (value - self.mean) / self.std
            patch = self.model.forward_features(value)["x_norm_patchtokens"]
            outputs.append(patch.float().cpu())
        return torch.cat(outputs)


def dino_metrics(generated: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    generated = F.normalize(generated.float(), dim=-1)
    source = F.normalize(source.float(), dim=-1)
    patch_cosine = (generated * source).sum(dim=-1).mean()
    generated_global = F.normalize(generated.mean(dim=1), dim=-1)
    source_global = F.normalize(source.mean(dim=1), dim=-1)
    global_cosine = (generated_global * source_global).sum(dim=-1).mean()
    generated_temporal = generated_global @ generated_global.T
    source_temporal = source_global @ source_global.T
    temporal_gram_mae = (generated_temporal - source_temporal).abs().mean()
    return {
        "dino_patch_cosine_to_source": patch_cosine.item(),
        "dino_global_cosine_to_source": global_cosine.item(),
        "dino_temporal_gram_mae": temporal_gram_mae.item(),
    }


def add_sheet_row(canvas: Image.Image, row: int, label: str, frames: np.ndarray) -> None:
    draw = ImageDraw.Draw(canvas)
    y = row * 264
    draw.rectangle((0, y, 150, y + 264), fill="white")
    draw.text((8, y + 10), label, fill="black")
    indices = np.rint(np.linspace(0, len(frames) - 1, 5)).astype(np.int64)
    for column, index in enumerate(indices):
        image = Image.fromarray(frames[index]).resize((320, 240))
        x = 150 + column * 320
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + 240, x + 320, y + 264), fill="white")
        draw.text((x + 6, y + 244), f"frame {int(index)}", fill="black")


def main() -> None:
    args = parse_args()
    if len(args.video) != len(args.label):
        raise ValueError("--video and --label counts must match")
    generated = [(label, read_video(path)) for label, path in zip(args.label, args.video)]
    frame_count = min(len(frames) for _, frames in generated)
    generated = [(label, frames[:frame_count]) for label, frames in generated]
    source = read_video(args.source, frame_count)

    encoder = DinoEncoder(args.dino_repo, args.device)
    source_features = encoder(source, args.batch_size)
    results: dict[str, dict[str, float | int]] = {}
    for label, frames in generated:
        metrics: dict[str, float | int] = {"frames": len(frames)}
        metrics.update(pixel_metrics(frames))
        features = encoder(frames, args.batch_size)
        metrics.update(dino_metrics(features, source_features))
        first = torch.from_numpy(frames[0]).float().div(255.0)
        source_first = torch.from_numpy(source[0]).float().div(255.0)
        mse = F.mse_loss(first, source_first).item()
        metrics["first_frame_psnr_to_source"] = -10.0 * math.log10(max(mse, 1e-12))
        results[label] = metrics

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    canvas = Image.new("RGB", (150 + 5 * 320, len(generated) * 264), "white")
    for row, (label, frames) in enumerate(generated):
        add_sheet_row(canvas, row, label, frames)
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.contact_sheet, quality=92)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
