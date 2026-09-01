#!/usr/bin/env python3
"""Prepare the 10-checkpoint x condition/no-condition evaluation matrix."""

import json
from pathlib import Path


SOURCE = Path(
    "/datassd/morka/cosmos-work/eval/"
    "robotwin2_balanced50_native_sam3d_hybrid_iter4000_ar93_overlap01_640x480_121f_20260829"
)
HOST_ROOT = Path(
    "/bdsys_hdd/datassd/morka/cosmos-work/eval/"
    "sam3d_complete106k_every500_balanced50_ar93_overlap01_640x480_121f_20260901"
)
CONTAINER_ROOT = Path(
    "/datassd/morka/cosmos-work/eval/"
    "sam3d_complete106k_every500_balanced50_ar93_overlap01_640x480_121f_20260901"
)
CHECKPOINT_ROOT = Path(
    "/datassd/morka/cosmos-sam3d-work/training-outputs/"
    "sam3d-v93-complete106k-frommix121k-iter5000-lr1e5-bs4-32gpu-5k-20260831/"
    "cosmos_predict_v2p5_sam3d/video2world/"
    "sam3d-v93-complete106k-frommix121k-iter5000-lr1e5-bs4-32gpu-5k-20260831/checkpoints"
)


def load_rows(variant: str) -> list[dict]:
    path = SOURCE / variant / "inputs" / "all.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50:
        raise SystemExit(f"{path}: expected 50 rows, got {len(rows)}")
    return rows


def main() -> None:
    matrix = []
    for iteration in range(500, 5001, 500):
        label = f"iter{iteration:05d}"
        checkpoint = CHECKPOINT_ROOT / f"iter_{iteration:09d}"
        host_checkpoint = Path("/bdsys_hdd") / checkpoint.relative_to("/")
        if not (host_checkpoint / "model").is_dir():
            raise SystemExit(f"missing checkpoint: {host_checkpoint}")
        for variant in ("with_condition", "no_condition"):
            rows = load_rows(variant)
            host_target = HOST_ROOT / label / variant
            container_target = CONTAINER_ROOT / label / variant
            inputs = host_target / "inputs"
            for directory in (inputs, host_target / "videos", host_target / "logs", host_target / "state"):
                directory.mkdir(parents=True, exist_ok=True)
            normalized = []
            shards = [[] for _ in range(4)]
            for index, original in enumerate(rows):
                row = dict(original)
                episode = int(Path(row["input_path"]).stem.removeprefix("episode"))
                row["name"] = f"complete106k_{label}_{variant}_episode{episode:06d}"
                row.update(
                    resolution="480,640",
                    num_output_frames=121,
                    num_steps=35,
                    guidance=7,
                    enable_autoregressive=True,
                    chunk_size=93,
                    chunk_overlap=1,
                )
                if variant == "with_condition":
                    condition = row.get("sam3d_condition_path")
                    if not condition:
                        raise SystemExit(f"missing condition path for episode {episode}")
                    host_condition = Path("/bdsys_hdd") / Path(condition).relative_to("/")
                    if not host_condition.is_file():
                        raise SystemExit(f"missing migrated condition: {host_condition}")
                else:
                    row.pop("sam3d_condition_path", None)
                normalized.append(row)
                shards[index % 4].append(row)
            (inputs / "all.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8"
            )
            for shard, shard_rows in enumerate(shards):
                (inputs / f"shard_{shard:02d}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard_rows), encoding="utf-8"
                )
            config = {
                "label": label,
                "variant": variant,
                "checkpoint_dir": str(checkpoint),
                "host_output": str(host_target / "videos"),
                "container_eval_root": str(container_target),
                "samples": 50,
                "inference": {
                    "frames": 121,
                    "resolution": "640x480",
                    "autoregressive": True,
                    "chunk_size": 93,
                    "chunk_overlap": 1,
                    "steps": 35,
                    "guidance": 7,
                    "fps": 24,
                },
                "sam3d_condition": variant == "with_condition",
                "required_modalities": ["mask", "mask_meta", "geometry", "shape", "pose"]
                if variant == "with_condition"
                else [],
                "teacher_as_condition": False,
            }
            (host_target / "run_config.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            matrix.append(config)
    (HOST_ROOT / "experiment_matrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"prepared {len(matrix)} experiments, {len(matrix) * 50} videos under {HOST_ROOT}")


if __name__ == "__main__":
    main()
