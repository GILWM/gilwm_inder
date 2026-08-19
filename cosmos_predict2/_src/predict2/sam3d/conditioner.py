# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Condition dataclass and conditioner config for Cosmos + SAM 3D."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.conditioner import BooleanFlag, GeneralConditioner, ReMapkey, TextAttr
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition


@dataclass(frozen=True)
class SAM3DVideo2WorldCondition(Video2WorldCondition):
    # Frozen SAM 3D encoder tokens.  F is a small set of uniformly sampled
    # teacher frames; frame zero is also used as the generation condition.
    sam3d_tokens_B_F_N_D: Optional[torch.Tensor] = None
    # Explicit SAM 3 instance masks and per-instance metadata.
    sam_mask_B_K_H_W: Optional[torch.Tensor] = None
    sam_mask_meta_B_K_D: Optional[torch.Tensor] = None
    # Dense first-frame geometry: depth, XYZ point map, normals, confidence.
    sam3d_geometry_B_C_H_W: Optional[torch.Tensor] = None
    # Native SAM 3D Objects stage-1 shape latents and decoded object pose
    # (translation xyz, quaternion wxyz, scale xyz).
    sam3d_shape_latents_B_K_N_D: Optional[torch.Tensor] = None
    sam3d_object_pose_B_K_D: Optional[torch.Tensor] = None
    # Action targets are supervision only. They pass through the conditioner
    # for device/context-parallel transport but are never added to DiT context.
    actions_B_T_D: Optional[torch.Tensor] = None
    action_valid_B: Optional[torch.Tensor] = None


class SAM3DVideo2WorldConditioner(GeneralConditioner):
    _OPTIONAL_SAM3D_KEYS = (
        "sam3d_tokens_B_F_N_D",
        "sam_mask_B_K_H_W",
        "sam_mask_meta_B_K_D",
        "sam3d_geometry_B_C_H_W",
        "sam3d_shape_latents_B_K_N_D",
        "sam3d_object_pose_B_K_D",
        "actions_B_T_D",
        "action_valid_B",
    )

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> SAM3DVideo2WorldCondition:
        # Official evaluation only has a prompt and the first frame.  The SAM
        # sidecars are optional at inference time, but GeneralConditioner
        # indexes every configured input before ReMapkey can preserve None.
        # Feed a tiny placeholder through the generic dropout machinery, then
        # restore the missing modalities to None so the DiT does not create a
        # synthetic SAM context or leak future-video teacher information.
        missing = [key for key in self._OPTIONAL_SAM3D_KEYS if batch.get(key) is None]
        if missing:
            batch = dict(batch)
            fps = batch["fps"]
            placeholder = torch.zeros((fps.shape[0], 1), dtype=torch.float32, device=fps.device)
            for key in missing:
                batch[key] = placeholder

        output = super()._forward(batch, override_dropout_rate)
        for key in missing:
            output[key] = None
        return SAM3DVideo2WorldCondition(**output)


SAM3DVideoPredictionConditioner: LazyDict = L(SAM3DVideo2WorldConditioner)(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
        use_empty_string=False,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
    sam3d_tokens=L(ReMapkey)(
        input_key="sam3d_tokens_B_F_N_D",
        output_key="sam3d_tokens_B_F_N_D",
        dropout_rate=0.1,
        dtype="bfloat16",
    ),
    sam_masks=L(ReMapkey)(
        input_key="sam_mask_B_K_H_W",
        output_key="sam_mask_B_K_H_W",
        dropout_rate=0.1,
        dtype="bfloat16",
    ),
    sam_mask_meta=L(ReMapkey)(
        input_key="sam_mask_meta_B_K_D",
        output_key="sam_mask_meta_B_K_D",
        dropout_rate=0.1,
        dtype="bfloat16",
    ),
    sam3d_geometry=L(ReMapkey)(
        input_key="sam3d_geometry_B_C_H_W",
        output_key="sam3d_geometry_B_C_H_W",
        dropout_rate=0.1,
        dtype="bfloat16",
    ),
    sam3d_shape_latents=L(ReMapkey)(
        input_key="sam3d_shape_latents_B_K_N_D",
        output_key="sam3d_shape_latents_B_K_N_D",
        dropout_rate=0.1,
        dtype="bfloat16",
    ),
    sam3d_object_pose=L(ReMapkey)(
        input_key="sam3d_object_pose_B_K_D",
        output_key="sam3d_object_pose_B_K_D",
        dropout_rate=0.1,
        dtype="bfloat16",
    ),
    actions=L(ReMapkey)(
        input_key="actions_B_T_D",
        output_key="actions_B_T_D",
        dropout_rate=0.0,
        dtype=None,
    ),
    action_valid=L(ReMapkey)(
        input_key="action_valid_B",
        output_key="action_valid_B",
        dropout_rate=0.0,
        dtype=None,
    ),
)
