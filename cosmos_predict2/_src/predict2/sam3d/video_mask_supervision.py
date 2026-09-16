# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Training-only temporal instance-mask supervision from frozen SAM 3 video."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoMaskLosses(NamedTuple):
    mask: torch.Tensor
    tracking: torch.Tensor


class TemporalVideoMaskHead(nn.Module):
    """Decode stable SAM-track slots from spatiotemporal DiT tokens.

    This head is used only while training.  Its targets may describe the full
    GT clip, but neither the head nor its targets participate in inference or
    in the DiT conditioning path.
    """

    def __init__(self, model_dim: int, hidden_dim: int, max_instances: int = 8) -> None:
        super().__init__()
        if min(model_dim, hidden_dim, max_instances) <= 0:
            raise ValueError("model_dim, hidden_dim, and max_instances must be positive")
        self.max_instances = int(max_instances)
        self.decoder = nn.Sequential(
            nn.LayerNorm(int(model_dim)),
            nn.Linear(int(model_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.max_instances),
        )

    def reset_parameters(self) -> None:
        for module in self.decoder.modules():
            if module is self.decoder:
                continue
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()

    @staticmethod
    def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.to(device=value.device, dtype=value.dtype)
        return (value * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(
        self,
        student_B_M_D: torch.Tensor,
        target_B_T_K_H_W: torch.Tensor,
        valid_B_T_K: torch.Tensor,
        *,
        latent_frames: int,
        spatial_shape: tuple[int, int],
    ) -> VideoMaskLosses:
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        expected_tokens = int(latent_frames) * height * width
        if student_B_M_D.ndim != 3 or student_B_M_D.shape[1] != expected_tokens:
            raise ValueError(
                f"student tokens must be [B,{expected_tokens},D] for T/H/W="
                f"{latent_frames}/{height}/{width}, got {tuple(student_B_M_D.shape)}"
            )
        if target_B_T_K_H_W.ndim != 5:
            raise ValueError(f"video mask target must be [B,T,K,H,W], got {tuple(target_B_T_K_H_W.shape)}")
        if tuple(target_B_T_K_H_W.shape[1:]) != (latent_frames, self.max_instances, height, width):
            raise ValueError(
                "video mask target shape mismatch: expected "
                f"[B,{latent_frames},{self.max_instances},{height},{width}], "
                f"got {tuple(target_B_T_K_H_W.shape)}"
            )
        if tuple(valid_B_T_K.shape) != tuple(target_B_T_K_H_W.shape[:3]):
            raise ValueError(
                f"video mask validity must be {tuple(target_B_T_K_H_W.shape[:3])}, got {tuple(valid_B_T_K.shape)}"
            )

        batch_size = student_B_M_D.shape[0]
        student = student_B_M_D.reshape(batch_size, latent_frames, height, width, -1)
        logits = self.decoder(student).permute(0, 1, 4, 2, 3).float()
        target = target_B_T_K_H_W.detach().to(device=logits.device, dtype=torch.float32)
        valid = valid_B_T_K.detach().to(device=logits.device, dtype=torch.bool)

        # Frame zero is visible to the video-to-world model.  Do not reward a
        # trivial reconstruction of that conditioning frame; supervise only
        # generated frames with the direct mask objective.
        direct_valid = valid.clone()
        direct_valid[:, 0] = False
        probabilities = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = probabilities * target + (1.0 - probabilities) * (1.0 - target)
        alpha_t = 0.75 * target + 0.25 * (1.0 - target)
        focal_per_mask = (alpha_t * (1.0 - p_t).square() * ce).mean(dim=(-2, -1))
        intersection = (probabilities * target).sum(dim=(-2, -1))
        denominator = probabilities.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice_per_mask = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        mask_loss = self._masked_mean(focal_per_mask + dice_per_mask, direct_valid)

        # Track motion as centroid displacement for each stable instance ID.
        # This term explicitly compares time t->t+1 with the same SAM track,
        # rather than aligning unrelated objects or global scene statistics.
        y_grid = torch.linspace(-1.0, 1.0, height, device=logits.device).view(1, 1, 1, height, 1)
        x_grid = torch.linspace(-1.0, 1.0, width, device=logits.device).view(1, 1, 1, 1, width)

        def centroid(value: torch.Tensor) -> torch.Tensor:
            mass = value.sum(dim=(-2, -1)).clamp_min(1.0e-4)
            center_x = (value * x_grid).sum(dim=(-2, -1)) / mass
            center_y = (value * y_grid).sum(dim=(-2, -1)) / mass
            return torch.stack((center_x, center_y), dim=-1)

        predicted_delta = centroid(probabilities)[:, 1:] - centroid(probabilities)[:, :-1]
        target_delta = centroid(target)[:, 1:] - centroid(target)[:, :-1]
        pair_valid = valid[:, 1:] & valid[:, :-1]
        tracking_per_pair = F.smooth_l1_loss(predicted_delta, target_delta, reduction="none").mean(dim=-1)
        tracking_loss = self._masked_mean(tracking_per_pair, pair_valid)
        return VideoMaskLosses(mask=mask_loss, tracking=tracking_loss)
