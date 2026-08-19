# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos Predict2.5 DiT with gated, multi-token SAM 3 / SAM 3D context.

Reason1 is represented by one flattened text token in Cosmos Predict2.5.  SAM
conditions must therefore not be squeezed into that position.  This module
builds a separate structured context sequence and lets every DiT block attend
to it through a zero-gated residual branch.  The branch reuses the pretrained
cross-attention projections, so it is parameter efficient and the all-zero
block gates make step zero exactly equivalent to the original Cosmos model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.sam3d.action_alignment import (
    CosmosActionExpertHead,
    TemporalActionAlignmentHead,
)


class SpatialConditionTokenizer(nn.Module):
    """Turn fixed-channel dense masks/geometry into a compact token sequence."""

    def __init__(self, in_channels: int, hidden_channels: int, output_channels: int, grid_size: int = 8):
        super().__init__()
        self.grid_size = grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, value_B_C_H_W: torch.Tensor) -> torch.Tensor:
        value_B_C_H_W = value_B_C_H_W.to(dtype=self.encoder[0].weight.dtype)
        tokens = self.encoder(value_B_C_H_W)
        tokens = F.adaptive_avg_pool2d(tokens, (self.grid_size, self.grid_size))
        return tokens.flatten(2).transpose(1, 2)


class ConditionProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, tokens_B_N_D: torch.Tensor) -> torch.Tensor:
        tokens_B_N_D = tokens_B_N_D.to(dtype=self.proj.weight.dtype)
        return self.proj(self.norm(tokens_B_N_D))


class ActionChunkConditioner(nn.Module):
    """Map frame-aligned robot actions to Cosmos temporal conditioning.

    A causal 4x VAE maps ``4N+1`` RGB frames to ``N+1`` latent frames.  The
    first latent is the observed condition frame; each later latent receives
    the four actions attached to its four newly generated RGB frames.  The
    stable MUSA path injects a bounded residual into the per-latent timestep
    embedding only; direct AdaLN residuals were unstable at large FSDP batch
    sizes.

    The MLP uses ordinary initialization and a small fixed residual scale so it
    receives gradients from the first optimization step without a learned
    zero gate.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        model_dim: int,
        actions_per_latent: int = 4,
        hidden_dim: Optional[int] = None,
        action_clip: Optional[float] = 10.0,
        residual_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if min(action_dim, model_dim, actions_per_latent) <= 0:
            raise ValueError("action_dim, model_dim, and actions_per_latent must be positive")
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.actions_per_latent = int(actions_per_latent)
        if action_clip is not None and float(action_clip) <= 0:
            raise ValueError(f"action_clip must be positive or None, got {action_clip}")
        self.action_clip = float(action_clip) if action_clip is not None else None
        if float(residual_scale) < 0:
            raise ValueError(f"residual_scale must be non-negative, got {residual_scale}")
        self.residual_scale = float(residual_scale)
        hidden_dim = int(hidden_dim) if hidden_dim is not None else self.model_dim * 4
        input_dim = self.action_dim * self.actions_per_latent
        activation = nn.GELU(approximate="tanh")
        self.timestep_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, self.model_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.timestep_embedder[0].reset_parameters()
        self.timestep_embedder[2].reset_parameters()

    def forward(
        self,
        actions_B_T_D: torch.Tensor,
        *,
        latent_frames: int,
        action_valid_B: Optional[torch.Tensor] = None,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        if actions_B_T_D.ndim != 3 or actions_B_T_D.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must be [B,T,{self.action_dim}], got {tuple(actions_B_T_D.shape)}"
            )
        if latent_frames < 1:
            raise ValueError(f"latent_frames must be positive, got {latent_frames}")
        expected_frames = 1 + (latent_frames - 1) * self.actions_per_latent
        if actions_B_T_D.shape[1] != expected_frames:
            raise ValueError(
                "Frame/action alignment mismatch: "
                f"{latent_frames} latent frames require {expected_frames} frame-aligned actions "
                f"(1 + ({latent_frames}-1)*{self.actions_per_latent}), "
                f"got {actions_B_T_D.shape[1]}"
            )

        batch_size = actions_B_T_D.shape[0]
        parameter = self.timestep_embedder[0].weight
        # action[t] is paired with RGB frame[t].  RGB frame zero is already
        # observed, so groups [1:5], [5:9], ... condition future latents.
        future = actions_B_T_D[:, 1:].to(device=parameter.device, dtype=parameter.dtype)
        if self.action_clip is not None:
            future = future.clamp(min=-self.action_clip, max=self.action_clip)
        chunks = future.reshape(batch_size, latent_frames - 1, -1)
        # Normalize each action residual before applying a small fixed scale.
        # A learned zero gate looks attractive for exact step-zero parity, but
        # its first non-zero Adam update is rank-locally unstable with MUSA
        # FSDP2 at large batch sizes.  A fixed residual scale keeps the signal
        # bounded while allowing the MLP to learn from the first step.
        timestep = self.timestep_embedder(chunks)
        timestep = F.layer_norm(timestep, (self.model_dim,)) * self.residual_scale
        timestep = F.pad(timestep, (0, 0, 1, 0))

        if action_valid_B is not None:
            valid = action_valid_B.to(device=timestep.device, dtype=timestep.dtype).reshape(-1, 1, 1)
            if valid.shape[0] != batch_size:
                raise ValueError(f"action_valid_B must contain {batch_size} values, got {valid.shape[0]}")
            timestep = timestep * valid

        if condition_video_input_mask_B_C_T_H_W is not None:
            mask = condition_video_input_mask_B_C_T_H_W
            if mask.shape[0] != batch_size or mask.shape[2] != latent_frames:
                raise ValueError(
                    "condition video mask must match action batch/latent time, "
                    f"got {tuple(mask.shape)} for B={batch_size}, T={latent_frames}"
                )
            generated = 1.0 - mask[:, :1, :, :1, :1].reshape(batch_size, latent_frames, 1)
            generated = generated.to(device=timestep.device, dtype=timestep.dtype)
            timestep = timestep * generated
        return timestep, None


class MinimalV1LVGSam3DDiT(MinimalV1LVGDiT):
    """MinimalV1LVGDiT with a structured frozen-teacher context branch."""

    _CONDITION_MODULE_MARKERS = (
        "sam3d_token_projector",
        "sam_mask_tokenizer",
        "sam_mask_projector",
        "sam_mask_meta_projector",
        "sam_geometry_tokenizer",
        "sam_geometry_projector",
        "sam3d_shape_projector",
        "sam3d_pose_projector",
        "sam3d_repa_projector",
        "action_supervision_head",
        "action_chunk_conditioner",
        "sam_modality_embeddings",
        "sam_context_block_gates",
    )

    def __init__(
        self,
        *args,
        sam3d_token_dim: int = 768,
        sam_mask_instances: int = 8,
        sam_mask_meta_dim: int = 8,
        sam_geometry_channels: int = 8,
        sam3d_shape_dim: int = 8,
        sam3d_pose_dim: int = 10,
        sam_condition_hidden_dim: int = 256,
        sam_condition_grid_size: int = 8,
        sam_condition_max_tokens: int = 256,
        sam_condition_modality_tokens: Tuple[int, int, int, int, int, int] = (64, 48, 8, 48, 80, 8),
        sam_condition_scale: float = 1.0,
        sam3d_repa_projection_dim: Optional[int] = None,
        sam3d_teacher_tokens_as_condition: bool = False,
        action_supervision_hidden_dim: Optional[int] = None,
        action_supervision_architecture: str = "lightweight",
        action_supervision_num_layers: int = 6,
        action_supervision_num_heads: int = 8,
        action_supervision_ffn_multiplier: int = 4,
        action_supervision_pool_grid: int = 2,
        action_dim: int = 14,
        action_conditioning_enabled: bool = False,
        action_conditioning_hidden_dim: Optional[int] = None,
        action_conditioning_actions_per_latent: int = 4,
        action_conditioning_clip: Optional[float] = 10.0,
        action_conditioning_scale: float = 0.01,
        **kwargs,
    ):
        crossattn_dim = int(kwargs.get("crossattn_emb_channels", 1024))
        model_channels = int(kwargs.get("model_channels", 2048))
        super().__init__(*args, **kwargs)

        self.sam_condition_max_tokens = sam_condition_max_tokens
        if len(sam_condition_modality_tokens) != 6 or any(count <= 0 for count in sam_condition_modality_tokens):
            raise ValueError("sam_condition_modality_tokens must contain six positive token budgets")
        self.sam_condition_modality_tokens = tuple(int(count) for count in sam_condition_modality_tokens)
        self.sam_condition_scale = sam_condition_scale
        # Legacy experiments exposed the full-video DINO teacher tokens to the
        # cross-attention branch.  Keep that behavior loadable, but new REPA
        # runs disable it: a teacher target must not also be an inference-time
        # input (especially when it contains future frames).
        if isinstance(sam3d_teacher_tokens_as_condition, str):
            normalized = sam3d_teacher_tokens_as_condition.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    "sam3d_teacher_tokens_as_condition must be true or false, "
                    f"got {sam3d_teacher_tokens_as_condition!r}"
                )
            self.sam3d_teacher_tokens_as_condition = normalized == "true"
        else:
            self.sam3d_teacher_tokens_as_condition = bool(sam3d_teacher_tokens_as_condition)
        self.sam3d_token_projector = ConditionProjector(sam3d_token_dim, crossattn_dim)
        self.sam_mask_tokenizer = SpatialConditionTokenizer(
            sam_mask_instances, sam_condition_hidden_dim, sam_condition_hidden_dim, sam_condition_grid_size
        )
        self.sam_mask_projector = ConditionProjector(sam_condition_hidden_dim, crossattn_dim)
        self.sam_mask_meta_projector = ConditionProjector(sam_mask_meta_dim, crossattn_dim)
        self.sam_geometry_tokenizer = SpatialConditionTokenizer(
            sam_geometry_channels, sam_condition_hidden_dim, sam_condition_hidden_dim, sam_condition_grid_size
        )
        self.sam_geometry_projector = ConditionProjector(sam_condition_hidden_dim, crossattn_dim)
        self.sam3d_shape_projector = ConditionProjector(sam3d_shape_dim, crossattn_dim)
        self.sam3d_pose_projector = ConditionProjector(sam3d_pose_dim, crossattn_dim)
        # Standard REPA uses an independent trainable student projection head.
        # Keep it optional so checkpoints made by the legacy relation-only
        # experiment remain structurally loadable without missing parameters.
        self.sam3d_repa_projector = (
            ConditionProjector(model_channels, int(sam3d_repa_projection_dim))
            if sam3d_repa_projection_dim is not None
            else None
        )
        # Kept structurally optional so all existing SAM3D checkpoints remain
        # loadable without missing action-head parameters. Training launchers
        # enable it only when action supervision has a non-zero weight.
        if action_supervision_architecture not in {"lightweight", "cosmos_action_expert"}:
            raise ValueError(
                "action_supervision_architecture must be lightweight or cosmos_action_expert, "
                f"got {action_supervision_architecture!r}"
            )
        self.action_supervision_architecture = action_supervision_architecture
        if action_supervision_hidden_dim is None:
            self.action_supervision_head = None
        elif action_supervision_architecture == "lightweight":
            self.action_supervision_head = TemporalActionAlignmentHead(
                model_dim=model_channels,
                action_dim=int(action_dim),
                hidden_dim=int(action_supervision_hidden_dim),
            )
        else:
            self.action_supervision_head = CosmosActionExpertHead(
                model_dim=model_channels,
                action_dim=int(action_dim),
                hidden_dim=int(action_supervision_hidden_dim),
                num_layers=int(action_supervision_num_layers),
                num_heads=int(action_supervision_num_heads),
                ffn_multiplier=int(action_supervision_ffn_multiplier),
                pool_grid=int(action_supervision_pool_grid),
            )
        if isinstance(action_conditioning_enabled, str):
            normalized = action_conditioning_enabled.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    "action_conditioning_enabled must be true or false, "
                    f"got {action_conditioning_enabled!r}"
                )
            self.action_conditioning_enabled = normalized == "true"
        else:
            self.action_conditioning_enabled = bool(action_conditioning_enabled)
        self.action_chunk_conditioner = (
            ActionChunkConditioner(
                action_dim=int(action_dim),
                model_dim=model_channels,
                actions_per_latent=int(action_conditioning_actions_per_latent),
                hidden_dim=action_conditioning_hidden_dim,
                action_clip=action_conditioning_clip,
                residual_scale=action_conditioning_scale,
            )
            if self.action_conditioning_enabled
            else None
        )

        # Modality identity is retained after concatenation.  One zero gate per
        # block follows the PAIWorld-style residual-adapter design and is kept
        # one-dimensional because FSDP2 rejects scalar parameters.
        self.sam_modality_embeddings = nn.Parameter(torch.empty(6, crossattn_dim))
        self.sam_context_block_gates = nn.Parameter(torch.zeros(self.num_blocks))
        self._init_condition_weights()

    def _init_condition_weights(self) -> None:
        for name in (
            "sam3d_token_projector",
            "sam_mask_tokenizer",
            "sam_mask_projector",
            "sam_mask_meta_projector",
            "sam_geometry_tokenizer",
            "sam_geometry_projector",
            "sam3d_shape_projector",
            "sam3d_pose_projector",
            "sam3d_repa_projector",
            "action_supervision_head",
            "action_chunk_conditioner",
        ):
            module = getattr(self, name, None)
            if module is None:
                continue
            if name == "action_chunk_conditioner":
                module.reset_parameters()
                continue
            for child in module.modules():
                if child is module:
                    continue
                reset_parameters = getattr(child, "reset_parameters", None)
                if reset_parameters is not None:
                    reset_parameters()
        modality_embeddings = getattr(self, "sam_modality_embeddings", None)
        if modality_embeddings is not None:
            nn.init.normal_(modality_embeddings, mean=0.0, std=0.02)
        block_gates = getattr(self, "sam_context_block_gates", None)
        if block_gates is not None:
            nn.init.zeros_(block_gates)

    def init_weights(self):
        # MiniTrainDIT.__init__ calls this before the condition modules exist;
        # build_net calls it again after meta tensors are materialized.
        super().init_weights()
        self._init_condition_weights()

    def fully_shard(self, mesh, **fsdp_kwargs):
        """Shard the large action conditioner as its own FSDP unit.

        Leaving its roughly 19M parameters in the top-level catch-all FSDP
        group produced peer-rank NaNs on the current MUSA FSDP2 build. Cosmos
        already gives every large transformer block its own FSDP unit; apply
        the same rule to the action MLP.
        """
        if self.action_chunk_conditioner is not None:
            fully_shard(
                self.action_chunk_conditioner,
                mesh=mesh,
                reshard_after_forward=True,
                **fsdp_kwargs,
            )
        return super().fully_shard(mesh, **fsdp_kwargs)

    @staticmethod
    def _flatten_teacher_frames(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 4:  # B, teacher_frame, token, channel
            return tokens.flatten(1, 2)
        if tokens.ndim != 3:
            raise ValueError(f"SAM 3D tokens must be [B,N,D] or [B,F,N,D], got {tuple(tokens.shape)}")
        return tokens

    def _project_modality(
        self,
        source_tokens: Optional[torch.Tensor],
        projector: nn.Module,
        modality_index: int,
        token_budget: int,
        output_dtype: torch.dtype,
        active_B: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if source_tokens is None or source_tokens.numel() == 0:
            return None
        if source_tokens.ndim != 3:
            raise ValueError(f"Condition tokens must be [B,N,D], got {tuple(source_tokens.shape)}")
        token_count = min(int(token_budget), source_tokens.shape[1])
        if source_tokens.shape[1] != token_count:
            source_tokens = F.adaptive_avg_pool1d(source_tokens.transpose(1, 2), token_count).transpose(1, 2)
        if active_B is None:
            active_B = source_tokens.detach().float().abs().amax(dim=(1, 2)) > 0
        projected = projector(source_tokens).to(dtype=output_dtype)
        modality = self.sam_modality_embeddings[modality_index].to(dtype=output_dtype).view(1, 1, -1)
        return (projected + modality) * active_B.to(dtype=output_dtype).view(-1, 1, 1)

    def _build_sam_context(
        self,
        output_dtype: torch.dtype,
        sam3d_tokens_B_F_N_D: Optional[torch.Tensor],
        sam_mask_B_K_H_W: Optional[torch.Tensor],
        sam_mask_meta_B_K_D: Optional[torch.Tensor],
        sam3d_geometry_B_C_H_W: Optional[torch.Tensor],
        sam3d_shape_latents_B_K_N_D: Optional[torch.Tensor],
        sam3d_object_pose_B_K_D: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        contexts: list[torch.Tensor] = []

        def append(value: Optional[torch.Tensor]) -> None:
            if value is not None:
                contexts.append(value)

        if self.sam3d_teacher_tokens_as_condition and sam3d_tokens_B_F_N_D is not None:
            tokens = self._flatten_teacher_frames(sam3d_tokens_B_F_N_D)
            append(
                self._project_modality(
                    tokens, self.sam3d_token_projector, 0, self.sam_condition_modality_tokens[0], output_dtype
                )
            )
        if sam_mask_B_K_H_W is not None:
            active = sam_mask_B_K_H_W.detach().float().abs().amax(dim=(1, 2, 3)) > 0
            append(
                self._project_modality(
                    self.sam_mask_tokenizer(sam_mask_B_K_H_W),
                    self.sam_mask_projector,
                    1,
                    self.sam_condition_modality_tokens[1],
                    output_dtype,
                    active,
                )
            )
        if sam_mask_meta_B_K_D is not None:
            append(
                self._project_modality(
                    sam_mask_meta_B_K_D,
                    self.sam_mask_meta_projector,
                    2,
                    self.sam_condition_modality_tokens[2],
                    output_dtype,
                )
            )
        if sam3d_geometry_B_C_H_W is not None:
            active = sam3d_geometry_B_C_H_W.detach().float().abs().amax(dim=(1, 2, 3)) > 0
            append(
                self._project_modality(
                    self.sam_geometry_tokenizer(sam3d_geometry_B_C_H_W),
                    self.sam_geometry_projector,
                    3,
                    self.sam_condition_modality_tokens[3],
                    output_dtype,
                    active,
                )
            )
        if sam3d_shape_latents_B_K_N_D is not None:
            if sam3d_shape_latents_B_K_N_D.ndim != 4:
                raise ValueError(
                    f"SAM 3D shape latents must be [B,K,N,D], got {tuple(sam3d_shape_latents_B_K_N_D.shape)}"
                )
            append(
                self._project_modality(
                    sam3d_shape_latents_B_K_N_D.flatten(1, 2),
                    self.sam3d_shape_projector,
                    4,
                    self.sam_condition_modality_tokens[4],
                    output_dtype,
                )
            )
        if sam3d_object_pose_B_K_D is not None:
            append(
                self._project_modality(
                    sam3d_object_pose_B_K_D,
                    self.sam3d_pose_projector,
                    5,
                    self.sam_condition_modality_tokens[5],
                    output_dtype,
                )
            )

        if not contexts:
            return None
        batch_sizes = {value.shape[0] for value in contexts}
        if len(batch_sizes) != 1:
            raise ValueError(f"SAM condition batch sizes do not match: {sorted(batch_sizes)}")
        context = torch.cat(contexts, dim=1)
        if context.shape[1] > self.sam_condition_max_tokens:
            context = F.adaptive_avg_pool1d(context.transpose(1, 2), self.sam_condition_max_tokens).transpose(1, 2)
        return context.contiguous()

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        sam3d_tokens_B_F_N_D: Optional[torch.Tensor] = None,
        sam_mask_B_K_H_W: Optional[torch.Tensor] = None,
        sam_mask_meta_B_K_D: Optional[torch.Tensor] = None,
        sam3d_geometry_B_C_H_W: Optional[torch.Tensor] = None,
        sam3d_shape_latents_B_K_N_D: Optional[torch.Tensor] = None,
        sam3d_object_pose_B_K_D: Optional[torch.Tensor] = None,
        actions_B_T_D: Optional[torch.Tensor] = None,
        action_valid_B: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        # Cosmos 2.5's Reason1 conditioning is stored as one flattened
        # 100352-dimensional token.  Project it through the native path, while
        # keeping SAM/SAM3D as a separate multi-token context branch.
        crossattn_already_projected = False
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
            crossattn_already_projected = True
        sam_context_emb = self._build_sam_context(
            output_dtype=crossattn_emb.dtype,
            sam3d_tokens_B_F_N_D=sam3d_tokens_B_F_N_D,
            sam_mask_B_K_H_W=sam_mask_B_K_H_W,
            sam_mask_meta_B_K_D=sam_mask_meta_B_K_D,
            sam3d_geometry_B_C_H_W=sam3d_geometry_B_C_H_W,
            sam3d_shape_latents_B_K_N_D=sam3d_shape_latents_B_K_N_D,
            sam3d_object_pose_B_K_D=sam3d_object_pose_B_K_D,
        )

        action_timestep = None
        action_adaln = None
        if self.action_chunk_conditioner is not None:
            if actions_B_T_D is None:
                raise RuntimeError(
                    "This checkpoint is action-conditioned, but inference/training did not provide actions_B_T_D"
                )
            action_timestep, action_adaln = self.action_chunk_conditioner(
                actions_B_T_D,
                latent_frames=x_B_C_T_H_W.shape[2],
                action_valid_B=action_valid_B,
                condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            )

        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            crossattn_already_projected=crossattn_already_projected,
            sam_context_emb=sam_context_emb,
            sam_context_block_gates=self.sam_context_block_gates * self.sam_condition_scale,
            t_embedding_addition_B_T_D=action_timestep,
            adaln_lora_addition_B_T_3D=action_adaln,
        )
