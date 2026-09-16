# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Time-aligned inverse-dynamics supervision for SAM3D video DiTs.

The legacy MLP head remains available for checkpoint compatibility.  The
Cosmos action expert below keeps the useful part of FlowWAM's design -- an
action-token transformer that reads a pyramid of video-DiT features -- while
using native Cosmos dimensions and a bounded spatiotemporal context.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
            raise ValueError(f"video tokens must be [B,M,{self.model_dim}], got {tuple(video_tokens_B_M_D.shape)}")
        if actions_B_T_D.ndim != 3 or actions_B_T_D.shape[-1] != self.action_dim:
            raise ValueError(f"actions must be [B,T,{self.action_dim}], got {tuple(actions_B_T_D.shape)}")
        if video_tokens_B_M_D.shape[0] != actions_B_T_D.shape[0]:
            raise ValueError("video/action batch sizes do not match")
        if video_tokens_B_M_D.shape[1] % latent_frames != 0:
            raise ValueError(
                f"DiT token count {video_tokens_B_M_D.shape[1]} is not divisible by latent_frames={latent_frames}"
            )

        spatial_tokens = video_tokens_B_M_D.shape[1] // latent_frames
        video_B_T_D = video_tokens_B_M_D.reshape(
            video_tokens_B_M_D.shape[0], latent_frames, spatial_tokens, self.model_dim
        ).mean(dim=2)
        action_indices = self.latent_action_indices(actions_B_T_D.shape[1], latent_frames, actions_B_T_D.device)
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


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a stable float32 sinusoidal embedding for arbitrary 1D values."""

    if dim < 2:
        raise ValueError(f"sinusoidal embedding dim must be >= 2, got {dim}")
    values = values.float().reshape(-1, 1)
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phases = values * frequencies.reshape(1, -1)
    embedding = torch.cat((phases.sin(), phases.cos()), dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        value_float = value.float()
        normalized = value_float * torch.rsqrt(value_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=dtype) * self.weight.to(dtype=dtype)


class _ActionAttention(nn.Module):
    """Multi-head attention with separate native query/context dimensions."""

    def __init__(self, dim: int, num_heads: int, context_dim: Optional[int] = None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        context_dim = dim if context_dim is None else int(context_dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.q_norm = _RMSNorm(dim)
        self.k_norm = _RMSNorm(dim)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = query if context is None else context
        q = self._split_heads(self.q_norm(self.q(query)))
        k = self._split_heads(self.k_norm(self.k(context)))
        v = self._split_heads(self.v(context))
        # MUSA's BF16 fused attention can produce non-finite softmax values on
        # real Cosmos activations even though the same shapes pass with random
        # tensors.  Action sequences are short after spatial pooling, so doing
        # only the attention kernel in FP32 is inexpensive and deterministic.
        attended = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(dtype=q.dtype)
        attended = attended.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], -1)
        return self.o(attended)


class _CosmosActionExpertBlock(nn.Module):
    """Action self-attention, video cross-attention and a gated FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.video_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attention = _ActionAttention(dim, num_heads)
        self.video_attention = _ActionAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        # One-dimensional gates are compatible with FSDP2.  A small initial
        # video gate prevents a random auxiliary head from shocking the video
        # backbone on its first optimizer step.
        self.self_gate = nn.Parameter(torch.ones(1))
        self.video_gate = nn.Parameter(torch.full((1,), 0.1))
        self.ffn_gate = nn.Parameter(torch.ones(1))

    @staticmethod
    def _scale(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return gate.to(device=value.device, dtype=value.dtype).reshape(1, 1, 1) * value

    def forward(self, action_tokens: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        action_tokens = action_tokens + self._scale(self.self_gate, self.self_attention(self.self_norm(action_tokens)))
        action_tokens = action_tokens + self._scale(
            self.video_gate,
            self.video_attention(self.video_norm(action_tokens), video_tokens),
        )
        action_tokens = action_tokens + self._scale(self.ffn_gate, self.ffn(self.ffn_norm(action_tokens)))
        return action_tokens


class CosmosActionExpertHead(nn.Module):
    """FlowWAM-style inverse-dynamics expert adapted to Cosmos token shapes.

    Unlike FlowWAM's action flow model, ground-truth actions are never used as
    input tokens here.  Fixed temporal queries read a pyramid of intermediate
    Cosmos features and predict one normalized 14D action per latent frame.
    This prevents target leakage while retaining per-layer video conditioning.

    Full ``T*H*W`` video attention is prohibitively expensive for 640x480
    training.  Each source feature is therefore reshaped back to its real 2D
    grid and adaptively pooled to ``pool_grid x pool_grid`` tokens per frame
    before a shared 2048 -> expert_dim projection.
    """

    uses_feature_pyramid = True

    def __init__(
        self,
        model_dim: int,
        action_dim: int = 14,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        pool_grid: int = 2,
        time_embedding_dim: int = 256,
    ):
        super().__init__()
        values = (model_dim, action_dim, hidden_dim, num_layers, num_heads, ffn_multiplier, pool_grid)
        if min(values) <= 0:
            raise ValueError(f"Cosmos action expert dimensions must be positive, got {values}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.model_dim = int(model_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.pool_grid = int(pool_grid)
        self.time_embedding_dim = int(time_embedding_dim)

        self.video_projector = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
        )
        self.query_bias = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.time_projector = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            _CosmosActionExpertBlock(
                self.hidden_dim,
                int(num_heads),
                self.hidden_dim * int(ffn_multiplier),
            )
            for _ in range(self.num_layers)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.action_target_encoder = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.hidden_dim),
        )

    @staticmethod
    def latent_action_indices(action_frames: int, latent_frames: int, device: torch.device) -> torch.Tensor:
        return TemporalActionAlignmentHead.latent_action_indices(action_frames, latent_frames, device)

    @staticmethod
    def _feature_map(block_index: int, block_count: int, feature_count: int) -> int:
        if feature_count < 1:
            raise ValueError("At least one video feature layer is required")
        if block_count == 1:
            return feature_count - 1
        return round(block_index * (feature_count - 1) / (block_count - 1))

    def _pool_video_feature(
        self,
        feature_B_M_D: torch.Tensor,
        *,
        latent_frames: int,
        spatial_shape: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        if feature_B_M_D.ndim != 3 or feature_B_M_D.shape[-1] != self.model_dim:
            raise ValueError(f"video features must be [B,M,{self.model_dim}], got {tuple(feature_B_M_D.shape)}")
        if feature_B_M_D.shape[1] % latent_frames != 0:
            raise ValueError(
                f"video token count {feature_B_M_D.shape[1]} is not divisible by latent_frames={latent_frames}"
            )
        spatial_tokens = feature_B_M_D.shape[1] // latent_frames
        if spatial_shape is None:
            # Compatibility fallback for callers that only know flattened
            # token counts.  Production training passes the exact patch grid.
            side = int(round(spatial_tokens**0.5))
            spatial_shape = (side, side) if side * side == spatial_tokens else (1, spatial_tokens)
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        if height * width != spatial_tokens:
            raise ValueError(
                f"spatial_shape={spatial_shape} contains {height * width} positions, "
                f"but each latent frame has {spatial_tokens} tokens"
            )
        feature = feature_B_M_D.reshape(feature_B_M_D.shape[0], latent_frames, height, width, self.model_dim)
        feature = feature.flatten(0, 1).permute(0, 3, 1, 2)
        feature = F.adaptive_avg_pool2d(feature, (self.pool_grid, self.pool_grid))
        feature = feature.flatten(2).transpose(1, 2)
        feature = feature.reshape(
            feature_B_M_D.shape[0], latent_frames * self.pool_grid * self.pool_grid, self.model_dim
        )
        parameter_dtype = self.video_projector[1].weight.dtype
        return self.video_projector(feature.to(dtype=parameter_dtype))

    @staticmethod
    def _batch_timestep(
        timesteps_B_T: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if timesteps_B_T is None:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        timesteps_B_T = timesteps_B_T.to(device=device)
        if timesteps_B_T.numel() == batch_size and timesteps_B_T.shape[0] != batch_size:
            timesteps_B_T = timesteps_B_T.reshape(batch_size, 1)
        if timesteps_B_T.shape[0] != batch_size:
            raise ValueError(f"timesteps batch {timesteps_B_T.shape[0]} does not match video batch {batch_size}")
        return timesteps_B_T.float().reshape(batch_size, -1).mean(dim=1)

    def forward(
        self,
        video_layer_features: torch.Tensor | Sequence[torch.Tensor],
        actions_B_T_D: torch.Tensor,
        *,
        latent_frames: int,
        valid_B: Optional[torch.Tensor] = None,
        spatial_shape: Optional[tuple[int, int]] = None,
        timesteps_B_T: Optional[torch.Tensor] = None,
    ) -> ActionAlignmentLosses:
        features = (
            [video_layer_features] if isinstance(video_layer_features, torch.Tensor) else list(video_layer_features)
        )
        if not features:
            raise ValueError("video_layer_features must contain at least one tensor")
        batch_size = features[0].shape[0]
        if actions_B_T_D.ndim != 3 or actions_B_T_D.shape != (
            batch_size,
            actions_B_T_D.shape[1],
            self.action_dim,
        ):
            raise ValueError(
                f"actions must be [B,T,{self.action_dim}] with B={batch_size}, got {tuple(actions_B_T_D.shape)}"
            )
        for feature in features:
            if feature.shape[0] != batch_size:
                raise ValueError("All video feature layers must have the same batch size")

        if valid_B is None:
            valid_B = torch.ones(batch_size, dtype=torch.bool, device=features[0].device)
        else:
            valid_B = valid_B.to(device=features[0].device, dtype=torch.bool).reshape(-1)
        if valid_B.shape[0] != batch_size:
            raise ValueError(f"valid_B must contain {batch_size} values, got {valid_B.shape}")
        if not torch.any(valid_B):
            zero = features[0].float().sum() * 0.0
            return ActionAlignmentLosses(zero, zero)

        features = [feature[valid_B] for feature in features]
        actions = actions_B_T_D.to(device=features[0].device)[valid_B]
        action_indices = self.latent_action_indices(actions.shape[1], latent_frames, actions.device)
        targets = actions.index_select(1, action_indices)
        contexts = [
            self._pool_video_feature(
                feature,
                latent_frames=latent_frames,
                spatial_shape=spatial_shape,
            )
            for feature in features
        ]

        positions = _sinusoidal_embedding(torch.arange(latent_frames, device=features[0].device), self.hidden_dim).to(
            dtype=contexts[0].dtype
        )
        timesteps = self._batch_timestep(
            timesteps_B_T,
            batch_size=batch_size,
            device=features[0].device,
        )[valid_B]
        timestep_embedding = self.time_projector(
            _sinusoidal_embedding(timesteps, self.time_embedding_dim).to(dtype=contexts[0].dtype)
        )
        action_tokens = (
            self.query_bias.to(dtype=contexts[0].dtype) + positions.unsqueeze(0) + timestep_embedding.unsqueeze(1)
        )
        action_tokens = action_tokens.expand(targets.shape[0], -1, -1)

        for block_index, block in enumerate(self.blocks):
            feature_index = self._feature_map(block_index, len(self.blocks), len(contexts))
            action_tokens = block(action_tokens, contexts[feature_index])

        hidden = self.output_norm(action_tokens)
        predicted_actions = self.action_head(hidden)
        prediction_loss = F.mse_loss(predicted_actions.float(), targets.float())
        encoded_targets = self.action_target_encoder(targets.to(dtype=self.action_target_encoder[1].weight.dtype))
        alignment_loss = (
            1.0
            - (
                F.normalize(hidden.float(), dim=-1, eps=1.0e-6)
                * F.normalize(encoded_targets.float(), dim=-1, eps=1.0e-6)
            ).sum(dim=-1)
        ).mean()
        if not torch.isfinite(prediction_loss) or not torch.isfinite(alignment_loss):
            tensors = {
                "targets": targets,
                "contexts": torch.stack(contexts),
                "hidden": hidden,
                "predicted_actions": predicted_actions,
                "encoded_targets": encoded_targets,
            }
            nonfinite = {name: int((~torch.isfinite(value.float())).sum().item()) for name, value in tensors.items()}
            raise FloatingPointError(
                "Non-finite Cosmos action loss: "
                f"prediction={prediction_loss.float().item()}, alignment={alignment_loss.float().item()}, "
                f"nonfinite_counts={nonfinite}"
            )
        return ActionAlignmentLosses(prediction_loss, alignment_loss)

    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
