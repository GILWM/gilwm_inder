# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos video2world training model with optional DINO representation alignment."""

from __future__ import annotations

from typing import Optional

import attrs
import torch
import torch.nn.functional as F

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.sam3d.conditioner import SAM3DVideo2WorldCondition


@attrs.define(slots=False)
class SAM3DVideo2WorldModelRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    # REPA stays opt-in until a motion-aware video teacher demonstrates a
    # consistent generation-quality win. DINO patch alignment alone improved
    # static semantics but did not improve actions in controlled video A/Bs.
    sam3d_repa_weight: float = 0.0
    sam3d_repa_layer: int = 7
    sam3d_repa_frames: int = 8
    sam3d_repa_tokens: int = 256
    sam3d_repa_spatial_anchors: int = 32
    sam3d_repa_temporal_anchors: int = 64
    # ``relation`` preserves old checkpoints. ``projected_cosine`` follows
    # standard REPA: a learned student projection and patch-wise cosine loss.
    sam3d_repa_mode: str = "relation"
    sam3d_repa_grid_size: int = 16
    sam3d_repa_temporal_weight: float = 0.25
    # Optional inverse-dynamics supervision. The prediction term is MSE in
    # normalized action space; the alignment term compares per-time latent and
    # encoded-action directions. Defaults preserve all historical runs.
    action_loss_weight: float = 0.0
    action_alignment_weight: float = 0.1
    action_feature_layer: int = 7
    # A feature pyramid is used only by the Cosmos action expert.  Empty keeps
    # the historical single-layer lightweight head and all old configs intact.
    action_feature_layers: tuple[int, ...] = ()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.sam3d_repa_weight < 0:
            raise ValueError("sam3d_repa_weight must be non-negative")
        if self.action_loss_weight < 0 or self.action_alignment_weight < 0:
            raise ValueError("action loss weights must be non-negative")
        if self.action_feature_layer < 0:
            raise ValueError("action_feature_layer must be non-negative")
        self.action_feature_layers = tuple(int(layer) for layer in self.action_feature_layers)
        if any(layer < 0 for layer in self.action_feature_layers):
            raise ValueError("action_feature_layers must contain only non-negative layer IDs")


def _resample_tokens(tokens_B_N_D: torch.Tensor, token_count: int) -> torch.Tensor:
    return F.adaptive_avg_pool1d(tokens_B_N_D.transpose(1, 2), token_count).transpose(1, 2)


def _sampled_relations(tokens_B_N_D: torch.Tensor, anchor_count: int) -> torch.Tensor:
    tokens_B_N_D = F.normalize(tokens_B_N_D.float(), dim=-1, eps=1e-6)
    anchor_count = min(anchor_count, tokens_B_N_D.shape[1])
    # Student and teacher must use the same spatial anchors.  Evenly spaced
    # anchors are deterministic across distributed ranks and cache versions.
    anchor_indices = (
        torch.linspace(0, tokens_B_N_D.shape[1] - 1, anchor_count, device=tokens_B_N_D.device).round().long()
    )
    anchors = tokens_B_N_D[:, anchor_indices]
    return torch.einsum("bnd,bkd->bnk", tokens_B_N_D, anchors)


class SAM3DVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    config: SAM3DVideo2WorldModelRectifiedFlowConfig

    def __init__(self, config: SAM3DVideo2WorldModelRectifiedFlowConfig):
        self._pending_sam3d_repa_loss: Optional[torch.Tensor] = None
        self._pending_sam3d_repa_spatial_loss: Optional[torch.Tensor] = None
        self._pending_sam3d_repa_temporal_loss: Optional[torch.Tensor] = None
        self._pending_action_prediction_loss: Optional[torch.Tensor] = None
        self._pending_action_alignment_loss: Optional[torch.Tensor] = None
        super().__init__(config)

    def add_lora(self, network: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        network = super().add_lora(network, *args, **kwargs)
        trainable_condition_parameters = 0
        markers = getattr(network, "_CONDITION_MODULE_MARKERS", None)
        if markers is None and hasattr(network, "base_model"):
            markers = getattr(network.base_model, "_CONDITION_MODULE_MARKERS", None)
        markers = markers or (
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
            "sam_modality_embeddings",
            "sam_context_block_gates",
        )
        for name, parameter in network.named_parameters():
            if any(marker in name for marker in markers):
                parameter.requires_grad_(True)
                parameter.data = parameter.data.float()
                trainable_condition_parameters += parameter.numel()
        log.info(f"Enabled {trainable_condition_parameters:,} trainable SAM condition-adapter parameters")
        return network

    def _action_supervision_head(self) -> torch.nn.Module:
        network = self.net
        head = getattr(network, "action_supervision_head", None)
        if head is None and hasattr(network, "base_model"):
            head = getattr(network.base_model, "action_supervision_head", None)
        if head is None:
            raise RuntimeError(
                "action_loss_weight > 0 requires model.config.net.action_supervision_hidden_dim to be set"
            )
        return head

    def _relation_alignment_loss(
        self,
        student_B_M_D: torch.Tensor,
        teacher_B_F_N_D: torch.Tensor,
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if teacher_B_F_N_D.ndim == 3:
            teacher_B_F_N_D = teacher_B_F_N_D.unsqueeze(1)
        if teacher_B_F_N_D.ndim != 4:
            raise ValueError(f"Teacher tokens must be [B,F,N,D], got {tuple(teacher_B_F_N_D.shape)}")

        if student_B_M_D.shape[1] % latent_frames != 0:
            raise ValueError(
                f"DiT token count {student_B_M_D.shape[1]} is not divisible by latent frames {latent_frames}"
            )
        spatial_tokens = student_B_M_D.shape[1] // latent_frames
        student = student_B_M_D.reshape(student_B_M_D.shape[0], latent_frames, spatial_tokens, student_B_M_D.shape[-1])

        frame_count = min(self.config.sam3d_repa_frames, student.shape[1], teacher_B_F_N_D.shape[1])
        student_frame_indices = (
            torch.linspace(0, student.shape[1] - 1, frame_count, device=student.device).round().long()
        )
        teacher_frame_indices = (
            torch.linspace(0, teacher_B_F_N_D.shape[1] - 1, frame_count, device=teacher_B_F_N_D.device).round().long()
        )
        student = student.index_select(1, student_frame_indices)
        teacher = teacher_B_F_N_D.index_select(1, teacher_frame_indices).detach().to(student.device)

        # Conditioner dropout zeros complete samples. They must not become
        # artificial all-zero teacher targets in the auxiliary objective.
        valid_B = teacher.float().abs().amax(dim=(1, 2, 3)) > 0
        if not torch.any(valid_B):
            zero = student.float().sum() * 0.0
            return zero, zero, zero
        student = student[valid_B]
        teacher = teacher[valid_B]

        batch_size = int(student.shape[0])
        token_count = self.config.sam3d_repa_tokens
        student_spatial = _resample_tokens(student.flatten(0, 1), token_count)
        teacher_spatial = _resample_tokens(teacher.flatten(0, 1), token_count)
        student_relations = _sampled_relations(student_spatial, self.config.sam3d_repa_spatial_anchors)
        teacher_relations = _sampled_relations(teacher_spatial, self.config.sam3d_repa_spatial_anchors)
        spatial_loss = F.smooth_l1_loss(student_relations, teacher_relations)

        student_clip = student_spatial.reshape(batch_size, frame_count * token_count, -1)
        teacher_clip = teacher_spatial.reshape(batch_size, frame_count * token_count, -1)
        student_temporal_relations = _sampled_relations(student_clip, self.config.sam3d_repa_temporal_anchors)
        teacher_temporal_relations = _sampled_relations(teacher_clip, self.config.sam3d_repa_temporal_anchors)
        temporal_loss = F.smooth_l1_loss(student_temporal_relations, teacher_temporal_relations)
        return spatial_loss + temporal_loss, spatial_loss, temporal_loss

    def _repa_projector(self) -> torch.nn.Module:
        network = self.net
        projector = getattr(network, "sam3d_repa_projector", None)
        if projector is None and hasattr(network, "base_model"):
            projector = getattr(network.base_model, "sam3d_repa_projector", None)
        if projector is None:
            raise RuntimeError(
                "projected_cosine REPA requires net.sam3d_repa_projection_dim to match the teacher feature dimension"
            )
        return projector

    def _projected_cosine_alignment_loss(
        self,
        student_B_M_D: torch.Tensor,
        teacher_B_F_N_D: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align clean-DINO patches to projected noisy-DiT patches.

        Student tokens are restored to their real 2D grid before pooling. The
        old implementation pooled the flattened H*W axis, which mixed row
        boundaries and did not preserve patch correspondence.
        """

        if teacher_B_F_N_D.ndim == 3:
            teacher_B_F_N_D = teacher_B_F_N_D.unsqueeze(1)
        if teacher_B_F_N_D.ndim != 4:
            raise ValueError(f"Teacher tokens must be [B,F,N,D], got {tuple(teacher_B_F_N_D.shape)}")

        patch_spatial = int(getattr(self.net, "patch_spatial", 2))
        student_h = latent_height // patch_spatial
        student_w = latent_width // patch_spatial
        expected_tokens = latent_frames * student_h * student_w
        if student_B_M_D.shape[1] != expected_tokens:
            raise ValueError(
                f"DiT token count {student_B_M_D.shape[1]} != T*H*W {latent_frames}*{student_h}*{student_w}"
            )
        student = student_B_M_D.reshape(
            student_B_M_D.shape[0], latent_frames, student_h, student_w, student_B_M_D.shape[-1]
        )

        frame_count = min(self.config.sam3d_repa_frames, student.shape[1], teacher_B_F_N_D.shape[1])
        student_indices = torch.linspace(0, student.shape[1] - 1, frame_count, device=student.device).round().long()
        teacher_indices = (
            torch.linspace(0, teacher_B_F_N_D.shape[1] - 1, frame_count, device=teacher_B_F_N_D.device).round().long()
        )
        student = student.index_select(1, student_indices)
        teacher = teacher_B_F_N_D.index_select(1, teacher_indices).detach().to(student.device)

        valid_B = teacher.float().abs().amax(dim=(1, 2, 3)) > 0
        if not torch.any(valid_B):
            zero = student.float().sum() * 0.0
            return zero, zero, zero
        student = student[valid_B]
        teacher = teacher[valid_B]
        batch_size = int(student.shape[0])

        grid = int(self.config.sam3d_repa_grid_size)
        student = student.flatten(0, 1).permute(0, 3, 1, 2)
        student = F.adaptive_avg_pool2d(student, (grid, grid)).flatten(2).transpose(1, 2)

        teacher_side = int(round(teacher.shape[2] ** 0.5))
        if teacher_side * teacher_side == teacher.shape[2]:
            teacher = teacher.flatten(0, 1).transpose(1, 2).reshape(-1, teacher.shape[-1], teacher_side, teacher_side)
            teacher = F.adaptive_avg_pool2d(teacher, (grid, grid)).flatten(2).transpose(1, 2)
        else:
            teacher = _resample_tokens(teacher.flatten(0, 1), grid * grid)

        projected = self._repa_projector()(student)
        projected = F.normalize(projected.float(), dim=-1, eps=1e-6)
        teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6)
        spatial_loss = (1.0 - (projected * teacher).sum(dim=-1)).mean()

        projected_frames = projected.reshape(batch_size, frame_count, grid * grid, -1)
        teacher_frames = teacher.reshape(batch_size, frame_count, grid * grid, -1)
        if frame_count < 2:
            temporal_loss = projected_frames.sum() * 0.0
        else:
            # A clip-level Gram matrix is easy to satisfy with nearly static
            # features.  Instead, supervise the direction of feature change at
            # every spatial cell and give moving teacher cells more weight.
            # A static student has zero delta and therefore cannot minimize
            # this term by simply preserving the first-frame appearance.
            student_delta = projected_frames[:, 1:] - projected_frames[:, :-1]
            teacher_delta = teacher_frames[:, 1:] - teacher_frames[:, :-1]
            teacher_motion = teacher_delta.detach().float().norm(dim=-1)
            student_delta = F.normalize(student_delta.float(), dim=-1, eps=1e-6)
            teacher_delta = F.normalize(teacher_delta.float(), dim=-1, eps=1e-6)
            delta_cosine = 1.0 - (student_delta * teacher_delta).sum(dim=-1)
            motion_mass = teacher_motion.sum()
            if motion_mass.item() <= 1e-6:
                temporal_loss = projected_frames.sum() * 0.0
            else:
                temporal_loss = (delta_cosine * teacher_motion).sum() / motion_mass
        total = spatial_loss + self.config.sam3d_repa_temporal_weight * temporal_loss
        return total, spatial_loss, temporal_loss

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: SAM3DVideo2WorldCondition,
    ) -> DenoisePrediction:
        condition_video_mask = None
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            channel_count = xt_B_C_T_H_W.shape[1]
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, channel_count, 1, 1, 1
            ).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                if timesteps_B_T.ndim == 1:
                    timesteps_B_T = timesteps_B_T.unsqueeze(0)

        collect_repa = (
            self.training and self.config.sam3d_repa_weight > 0 and condition.sam3d_tokens_B_F_N_D is not None
        )
        collect_action = self.training and self.config.action_loss_weight > 0
        if collect_action and condition.actions_B_T_D is None:
            raise RuntimeError(
                "Action supervision is enabled but the batch has no actions_B_T_D. "
                "Set dataloader action_hdf5_root/action_required or disable action_loss_weight."
            )
        action_head = self._action_supervision_head() if collect_action else None
        use_action_pyramid = bool(action_head is not None and getattr(action_head, "uses_feature_pyramid", False))
        configured_action_layers = tuple(getattr(self.config, "action_feature_layers", ()))
        action_feature_ids = (
            list(configured_action_layers)
            if use_action_pyramid and configured_action_layers
            else [self.config.action_feature_layer]
        )
        feature_ids = sorted(
            {
                *([self.config.sam3d_repa_layer] if collect_repa else []),
                *(action_feature_ids if collect_action else []),
            }
        )
        net_result = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            intermediate_feature_ids=feature_ids or None,
            **condition.to_dict(),
        )
        feature_by_id: dict[int, torch.Tensor] = {}
        if feature_ids:
            net_output_B_C_T_H_W, intermediate_features = net_result
            if len(intermediate_features) != len(feature_ids):
                raise RuntimeError(
                    f"Requested DiT features {feature_ids}, received {len(intermediate_features)} tensors"
                )
            feature_by_id = dict(zip(feature_ids, intermediate_features))
        else:
            net_output_B_C_T_H_W = net_result

        if collect_repa:
            repa_feature = feature_by_id[self.config.sam3d_repa_layer]
            if self.config.sam3d_repa_mode == "relation":
                losses = self._relation_alignment_loss(
                    repa_feature,
                    condition.sam3d_tokens_B_F_N_D,
                    latent_frames=xt_B_C_T_H_W.shape[2],
                )
            elif self.config.sam3d_repa_mode == "projected_cosine":
                losses = self._projected_cosine_alignment_loss(
                    repa_feature,
                    condition.sam3d_tokens_B_F_N_D,
                    latent_frames=xt_B_C_T_H_W.shape[2],
                    latent_height=xt_B_C_T_H_W.shape[3],
                    latent_width=xt_B_C_T_H_W.shape[4],
                )
            else:
                raise ValueError(f"Unsupported sam3d_repa_mode: {self.config.sam3d_repa_mode}")
            (
                self._pending_sam3d_repa_loss,
                self._pending_sam3d_repa_spatial_loss,
                self._pending_sam3d_repa_temporal_loss,
            ) = losses
        else:
            self._pending_sam3d_repa_loss = None
            self._pending_sam3d_repa_spatial_loss = None
            self._pending_sam3d_repa_temporal_loss = None

        if collect_action:
            action_features = [feature_by_id[layer] for layer in action_feature_ids]
            action_input = action_features if use_action_pyramid else action_features[-1]
            action_network = self.net.base_model if hasattr(self.net, "base_model") else self.net
            patch_spatial = int(getattr(action_network, "patch_spatial", 2))
            action_kwargs = dict(
                latent_frames=xt_B_C_T_H_W.shape[2],
                valid_B=condition.action_valid_B,
            )
            if use_action_pyramid:
                action_kwargs.update(
                    spatial_shape=(
                        xt_B_C_T_H_W.shape[3] // patch_spatial,
                        xt_B_C_T_H_W.shape[4] // patch_spatial,
                    ),
                    timesteps_B_T=timesteps_B_T,
                )
            action_losses = action_head(action_input, condition.actions_B_T_D, **action_kwargs)
            self._pending_action_prediction_loss = action_losses.prediction
            self._pending_action_alignment_loss = action_losses.alignment
        else:
            self._pending_action_prediction_loss = None
            self._pending_action_alignment_loss = None
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )
        return net_output_B_C_T_H_W

    def forward(self, data_batch: dict[str, torch.Tensor]):
        self._pending_sam3d_repa_loss = None
        self._pending_sam3d_repa_spatial_loss = None
        self._pending_sam3d_repa_temporal_loss = None
        self._pending_action_prediction_loss = None
        self._pending_action_alignment_loss = None
        output_batch, diffusion_loss = super().forward(data_batch)
        repa_loss = self._pending_sam3d_repa_loss
        if repa_loss is None:
            repa_loss = diffusion_loss.new_zeros(())
        action_prediction_loss = self._pending_action_prediction_loss
        if action_prediction_loss is None:
            action_prediction_loss = diffusion_loss.new_zeros(())
        action_alignment_loss = self._pending_action_alignment_loss
        if action_alignment_loss is None:
            action_alignment_loss = diffusion_loss.new_zeros(())
        action_supervision_loss = action_prediction_loss + self.config.action_alignment_weight * action_alignment_loss
        total_loss = (
            diffusion_loss
            + self.config.sam3d_repa_weight * repa_loss
            + self.config.action_loss_weight * action_supervision_loss
        )
        output_batch["diffusion_loss"] = diffusion_loss
        output_batch["sam3d_repa_loss"] = repa_loss
        if self._pending_sam3d_repa_spatial_loss is not None:
            output_batch["sam3d_repa_spatial_loss"] = self._pending_sam3d_repa_spatial_loss
        if self._pending_sam3d_repa_temporal_loss is not None:
            output_batch["sam3d_repa_temporal_loss"] = self._pending_sam3d_repa_temporal_loss
        output_batch["action_prediction_loss"] = action_prediction_loss
        output_batch["action_alignment_loss"] = action_alignment_loss
        output_batch["action_supervision_loss"] = action_supervision_loss
        output_batch["edm_loss"] = total_loss
        return output_batch, total_loss
