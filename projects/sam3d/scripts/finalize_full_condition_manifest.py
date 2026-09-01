#!/usr/bin/env python3
"""Atomically publish a manifest containing every validated full-condition sample."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o664)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--native-index", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--expected-source-count", type=int, default=121_099)
    args = parser.parse_args()

    manifest = args.dataset_root / "manifest.jsonl"
    backup = args.dataset_root / f"manifest.all_{args.expected_source_count}.jsonl"
    filtered = args.dataset_root / "manifest.full_conditions.jsonl"
    summary_path = args.dataset_root / "manifest.full_conditions.summary.json"
    active_summary_path = args.dataset_root / "manifest.summary.json"
    source_summary_backup = args.dataset_root / f"manifest.summary.all_{args.expected_source_count}.json"

    if not backup.is_file():
        shutil.copy2(manifest, backup)
        os.chmod(backup, 0o664)
    if active_summary_path.is_file() and not source_summary_backup.is_file():
        shutil.copy2(active_summary_path, source_summary_backup)
        os.chmod(source_summary_backup, 0o664)
    source = backup

    native_samples = [
        line.strip().replace("\\", "/")
        for line in args.native_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    native_set = set(native_samples)
    if len(native_samples) != len(native_set):
        raise ValueError("Native condition index contains duplicate entries")

    source_count = 0
    matched: set[str] = set()
    output_lines: list[str] = []
    batch_counts: dict[str, int] = {}
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        source_count += 1
        record = json.loads(raw_line)
        sample = str(record["sample_dir"]).replace("\\", "/")
        if sample not in native_set:
            continue
        if sample in matched:
            raise ValueError(f"Duplicate sample_dir in source manifest: {sample}")
        condition_path = args.cache_root / sample / "condition.pt"
        sidecar_path = args.sidecar_root / sample / "sam3d_objects.pt"
        if not condition_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"Validated condition disappeared: {sample}")
        record["sam3d_condition_path"] = str(condition_path)
        record["sam3d_objects_path"] = str(sidecar_path)
        record["sam3d_complete"] = True
        output_lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        matched.add(sample)
        batch = str(record.get("batch", "unknown"))
        batch_counts[batch] = batch_counts.get(batch, 0) + 1

    if source_count != args.expected_source_count:
        raise ValueError(f"Source manifest count={source_count}, expected={args.expected_source_count}")
    if matched != native_set:
        missing = sorted(native_set - matched)[:20]
        raise ValueError(f"Native index has {len(native_set - matched)} samples absent from manifest: {missing}")

    payload = "\n".join(output_lines) + ("\n" if output_lines else "")
    atomic_write(filtered, payload)
    atomic_write(manifest, payload)
    summary = {
        "schema": "sam3d_full_condition_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source),
        "source_records": source_count,
        "validated_native_index": str(args.native_index),
        "validated_condition_records": len(output_lines),
        "excluded_without_complete_condition": source_count - len(output_lines),
        "batch_counts": dict(sorted(batch_counts.items())),
        "published_manifest": str(manifest),
        "canonical_filtered_manifest": str(filtered),
        "teacher_tokens_required": False,
    }
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    atomic_write(summary_path, summary_payload)
    atomic_write(active_summary_path, summary_payload)
    print(
        "FULL_CONDITION_MANIFEST_READY",
        f"source={source_count}",
        f"published={len(output_lines)}",
        f"excluded={source_count - len(output_lines)}",
        f"manifest={manifest}",
        f"backup={backup}",
        flush=True,
    )


if __name__ == "__main__":
    main()
