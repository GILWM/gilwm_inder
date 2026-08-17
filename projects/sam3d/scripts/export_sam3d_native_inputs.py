#!/usr/bin/env python3
"""Build a compact, local-relay-friendly input tree for native SAM 3D stage-1."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mask.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_one(payload: tuple[dict[str, Any], str, str, str, int]) -> tuple[str, dict[str, Any] | None, str]:
    record, dataset_root_s, cache_root_s, output_root_s, jpeg_quality = payload
    dataset_root = Path(dataset_root_s)
    cache_root = Path(cache_root_s)
    output_root = Path(output_root_s)
    sample = Path(record.get("sample_dir", f"{record['batch']}/{record['question_id']}"))
    condition_path = cache_root / sample / "condition.pt"
    if not condition_path.is_file():
        return "missing_cache", None, str(sample)
    try:
        condition = torch.load(condition_path, map_location="cpu", weights_only=True)
    except TypeError:
        condition = torch.load(condition_path, map_location="cpu")
    valid_count = int(torch.as_tensor(condition.get("mask_valid_count", 0)).item())
    if valid_count < 1:
        return "no_mask", None, str(sample)
    masks = torch.as_tensor(condition["sam_masks"])
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    compact_mask = (masks[:1] > 0.5).to(torch.uint8).contiguous()
    if not torch.count_nonzero(compact_mask):
        return "empty_mask", None, str(sample)

    first_source = dataset_root / record["first_frame"]
    if not first_source.is_file():
        return "missing_first_frame", None, str(sample)
    first_relative = sample / "first.jpg"
    first_destination = output_root / "dataset" / first_relative
    first_destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(first_source) as image:
        image.convert("RGB").save(first_destination, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)

    atomic_torch_save(
        {
            "sam_masks": compact_mask,
            "mask_valid_count": torch.tensor(1, dtype=torch.int64),
        },
        output_root / "mask-cache" / sample / "condition.pt",
    )
    compact_record = dict(record)
    compact_record["first_frame"] = first_relative.as_posix()
    return "ok", compact_record, sample.as_posix()


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        (record, str(args.dataset_root), str(args.cache_root), str(args.output_root), args.jpeg_quality)
        for record in records
    ]
    counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (status, record, sample) in enumerate(executor.map(process_one, payloads), 1):
            counts[status] = counts.get(status, 0) + 1
            if record is not None:
                kept.append(record)
            if index % 250 == 0 or index == len(payloads):
                print(f"EXPORT position={index}/{len(payloads)} kept={len(kept)} counts={counts}", flush=True)

    manifest_path = args.output_root / "dataset" / "manifest.jsonl"
    index_path = args.output_root / "native-input-index.txt"
    manifest_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept), encoding="utf-8")
    index_path.write_text("".join(Path(record["sample_dir"]).as_posix() + "\n" for record in kept), encoding="utf-8")
    print(
        f"SAM3D_NATIVE_INPUT_EXPORT_OK records={len(records)} kept={len(kept)} "
        f"manifest={manifest_path} index={index_path} counts={counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
