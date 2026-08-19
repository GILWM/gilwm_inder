#!/usr/bin/env python3
"""Summarize and gate Cosmos action-conditioned training loss trends."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ITERATION = re.compile(r"Iteration\s+(?P<iteration>\d+):(?P<body>.*)")
METRIC = re.compile(r"(?P<name>[A-Za-z0-9_]+):\s*(?P<value>[-+0-9.eE]+)")
REQUIRED_METRICS = (
    "Loss",
    "diffusion_loss",
    "action_prediction_loss",
    "action_alignment_loss",
    "action_supervision_loss",
    "Time",
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--require-iterations", type=int, default=0)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--min-action-improvement", type=float, default=0.05)
    parser.add_argument(
        "--require-action-supervision",
        action="store_true",
        help="Require the optional inverse-dynamics loss to improve. The action-conditioned generator normally disables it.",
    )
    parser.add_argument("--max-diffusion-ratio", type=float, default=2.0)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    for line in args.log.read_text(encoding="utf-8").splitlines():
        match = ITERATION.search(line)
        if match is None:
            continue
        row: dict[str, float | int] = {"iteration": int(match.group("iteration"))}
        row.update({item.group("name"): float(item.group("value")) for item in METRIC.finditer(match.group("body"))})
        missing = [name for name in REQUIRED_METRICS if name not in row]
        if missing:
            raise ValueError(f"Iteration {row['iteration']} is missing metrics: {missing}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No iteration metrics found in {args.log}")

    failures: list[str] = []
    if args.require_iterations and int(rows[-1]["iteration"]) < args.require_iterations:
        failures.append(f"only reached iteration {rows[-1]['iteration']}, require {args.require_iterations}")
    for row in rows:
        for name in REQUIRED_METRICS:
            if not math.isfinite(float(row[name])):
                failures.append(f"iteration {row['iteration']} has non-finite {name}={row[name]}")

    window = min(args.window, len(rows) // 2)
    if window < 1:
        raise ValueError("At least two iterations are required for a trend")
    first = rows[:window]
    last = rows[-window:]
    first_means = {name: mean([float(row[name]) for row in first]) for name in REQUIRED_METRICS}
    last_means = {name: mean([float(row[name]) for row in last]) for name in REQUIRED_METRICS}
    action_ratio = last_means["action_prediction_loss"] / max(first_means["action_prediction_loss"], 1.0e-12)
    diffusion_ratio = last_means["diffusion_loss"] / max(first_means["diffusion_loss"], 1.0e-12)
    if (
        args.require_action_supervision
        and first_means["action_prediction_loss"] <= 1.0e-12
    ):
        failures.append("action supervision was required but action_prediction_loss is identically zero")
    if (
        args.require_action_supervision
        and len(rows) >= args.window * 2
        and action_ratio > 1.0 - args.min_action_improvement
    ):
        failures.append(
            f"action prediction did not improve by {args.min_action_improvement:.1%}: ratio={action_ratio:.4f}"
        )
    if diffusion_ratio > args.max_diffusion_ratio:
        failures.append(
            f"diffusion loss ratio {diffusion_ratio:.4f} exceeds {args.max_diffusion_ratio:.4f}"
        )

    report = {
        "log": str(args.log),
        "iterations_logged": len(rows),
        "last_iteration": int(rows[-1]["iteration"]),
        "window": window,
        "first_window_means": first_means,
        "last_window_means": last_means,
        "action_prediction_ratio_last_over_first": action_ratio,
        "diffusion_ratio_last_over_first": diffusion_ratio,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
