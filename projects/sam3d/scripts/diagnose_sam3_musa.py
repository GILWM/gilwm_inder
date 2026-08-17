#!/usr/bin/env python3
"""Report SAM3 parameter dtypes and the exact failing module on MUSA."""

from __future__ import annotations

import argparse
import collections
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()

    import torch_musa  # noqa: F401
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device="musa:0",
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=False,
        compile=False,
    )
    parameter_dtypes = collections.Counter(str(parameter.dtype) for parameter in model.parameters())
    buffer_dtypes = collections.Counter(str(buffer.dtype) for buffer in model.buffers())
    mixed_linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None and module.weight.dtype != module.bias.dtype:
            mixed_linears.append((name, str(module.weight.dtype), str(module.bias.dtype)))
    print(
        "SAM3_MUSA_DTYPE_REPORT",
        f"parameter_dtypes={dict(parameter_dtypes)}",
        f"buffer_dtypes={dict(buffer_dtypes)}",
        f"mixed_linears={mixed_linears[:20]}",
        f"autocast_musa={torch.is_autocast_enabled('musa')}",
    )
    if not args.forward:
        return

    last_module = {"name": "<none>"}

    def hook(name: str):
        def capture(_module, inputs):
            input_dtypes = [str(value.dtype) for value in inputs if torch.is_tensor(value)]
            last_module["name"] = f"{name} inputs={input_dtypes}"

        return capture

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        processor = Sam3Processor(model, device="musa:0", confidence_threshold=0.25)
        state = processor.set_image(Image.new("RGB", (640, 480), color=(127, 127, 127)))
        print("SAM3_MUSA_IMAGE_FORWARD_OK")
        if args.text:
            output = processor.set_text_prompt("robot", state)
            print(
                "SAM3_MUSA_TEXT_FORWARD_OK",
                f"masks={tuple(output['masks'].shape)}",
                f"scores={tuple(output['scores'].shape)}",
            )
    except Exception:
        print(f"SAM3_MUSA_FORWARD_FAILED last_module={last_module['name']}")
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
