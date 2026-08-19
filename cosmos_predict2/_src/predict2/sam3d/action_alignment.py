# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight time-aligned action supervision for SAM3D video DiTs."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionAlignmentLosses(NamedTuple):
    prediction: torch.Tensor
    alignment: torch.Tensor


class TemporalActionAlignmentHead(nn.Module):
    """Predict actions from DiT tokens and align their per-time representations.

    The video DiT emits ``T * H * W`` tokens.  Spatial averaging produces one
    latent vector ``z_t`` per latent frame.  For causal 4x temporal VAEs and
    the usual ``4N+1`` clips, uniform action selection maps exactly to source
    frames ``0, 4, 8, ...`` and therefore retains the special first frame.
    """

    def __init__(self, model_dim: int, action_dim: int = 14, hidden_dim: int = 512):
        super().__init__()
        if min(model_dim, action_dim, hidden_dim) <= 0:
            raise ValueError("model_dim, action_dim, and hidden_dim must be positive")
        self.model_dim = int(model_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        # The encoder ends in d_model as requested by the action-alignment
        # design, while a bottleneck keeps the adapter lightweight.
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.model_dim),
        )
        self.video_projector = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.model_dim),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()

    @staticmethod
    def latent_action_indices(action_frames: int, latent_frames: int, device: torch.device) -> torch.Tensor:
        if action_frames < 1 or latent_frames < 1:
            raise ValueError("action_frames and latent_frames must be positive")
        return torch.linspace(0, action_frames - 1, latent_frames, device=device).round().long()

    def forward(
        self,
        video_tokens_B_M_D: torch.Tensor,
        actions_B_T_D: torch.Tensor,
        *,
        latent_frames: int,
        valid_B: Optional[torch.Tensor] = None,
    ) -> ActionAlignmentLosses:
        if video_tokens_B_M_D.ndim != 3 or video_tokens_B_M_D.shape[-1] != self.model_dim:
            raise ValueError(
                f"video tokens must be [B,M,{self.model_dim}], got {tuple(video_tokens_B_M_D.shape)}"
            )
        if actions_B_T_D.ndim != 3 or actions_B_T_D.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must be [B,T,{self.action_dim}], got {tuple(actions_B_T_D.shape)}"
            )
        if video_tokens_B_M_D.shape[0] != actions_B_T_D.shape[0]:
            raise ValueError("video/action batch sizes do not match")
        if video_tokens_B_M_D.shape[1] % latent_frames != 0:
            raise ValueError(
                f"DiT token count {video_tokens_B_M_D.shape[1]} is not divisible by "
                f"latent_frames={latent_frames}"
            )

        spatial_tokens = video_tokens_B_M_D.shape[1] // latent_frames
        video_B_T_D = video_tokens_B_M_D.reshape(
            video_tokens_B_M_D.shape[0], latent_frames, spatial_tokens, self.model_dim
        ).mean(dim=2)
        action_indices = self.latent_action_indices(
            actions_B_T_D.shape[1], latent_frames, actions_B_T_D.device
        )
        actions_B_T_D = actions_B_T_D.index_select(1, action_indices)

        if valid_B is None:
            valid_B = torch.ones(video_B_T_D.shape[0], dtype=torch.bool, device=video_B_T_D.device)
        else:
            valid_B = valid_B.to(device=video_B_T_D.device, dtype=torch.bool).reshape(-1)
        if valid_B.shape[0] != video_B_T_D.shape[0]:
            raise ValueError(f"valid_B must contain {video_B_T_D.shape[0]} values, got {valid_B.shape}")
        if not torch.any(valid_B):
            zero = video_B_T_D.float().sum() * 0.0
            return ActionAlignmentLosses(zero, zero)

        video_B_T_D = video_B_T_D[valid_B]
        actions_B_T_D = actions_B_T_D.to(video_B_T_D.device)[valid_B]
        parameter_dtype = self.action_head[1].weight.dtype
        video_input = video_B_T_D.to(dtype=parameter_dtype)
        action_input = actions_B_T_D.to(dtype=parameter_dtype)

        predicted_actions = self.action_head(video_input)
        prediction_loss = F.mse_loss(predicted_actions.float(), actions_B_T_D.float())

        projected_video = F.normalize(self.video_projector(video_input).float(), dim=-1, eps=1.0e-6)
        encoded_actions = F.normalize(self.action_encoder(action_input).float(), dim=-1, eps=1.0e-6)
        alignment_loss = (1.0 - (projected_video * encoded_actions).sum(dim=-1)).mean()
        return ActionAlignmentLosses(prediction_loss, alignment_loss)
