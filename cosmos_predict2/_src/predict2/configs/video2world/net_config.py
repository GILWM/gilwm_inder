# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.networks.sam3d_conditioned_dit import MinimalV1LVGSam3DDiT
from cosmos_predict2._src.predict2.networks.wan2pt1 import WanModel

WAN2PT1_1PT3B: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=36,  # 16 (VAE channels) + 20 (image conditioning)
    model_type="i2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
    attention_backend="minimal_a2a",
)

WAN2PT1_14B: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
    attention_backend="minimal_a2a",
)

COSMOS_V1_7B_NET_MININET: LazyDict = L(MinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_2B_NET_MININET.model_channels = 2048
COSMOS_V1_2B_NET_MININET.num_heads = 16
COSMOS_V1_2B_NET_MININET.num_blocks = 28
COSMOS_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_14B_NET_MININET.model_channels = 5120
COSMOS_V1_14B_NET_MININET.num_heads = 40
COSMOS_V1_14B_NET_MININET.num_blocks = 36
COSMOS_V1_14B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_2B_SAM3D_NET: LazyDict = L(MinimalV1LVGSam3DDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
    sam3d_token_dim=768,
    sam_mask_instances=8,
    sam_mask_meta_dim=8,
    sam_geometry_channels=8,
    sam3d_shape_dim=8,
    sam3d_pose_dim=10,
    sam_condition_hidden_dim=256,
    sam_condition_grid_size=8,
    sam_condition_max_tokens=256,
    sam_condition_modality_tokens=(64, 48, 8, 48, 80, 8),
    # Disabled in the legacy relation-loss experiment. New projected-REPA
    # launches override this to the frozen DINO teacher width (768).
    sam3d_repa_projection_dim=None,
    # Optional lightweight inverse-dynamics head. Kept None for checkpoint
    # compatibility; action-supervised launches override it (typically 512).
    action_supervision_hidden_dim=None,
    action_supervision_architecture="lightweight",
    action_supervision_num_layers=6,
    action_supervision_num_heads=8,
    action_supervision_ffn_multiplier=4,
    action_supervision_pool_grid=2,
    action_dim=14,
    # True action-conditioned generation.  Disabled for old checkpoints and
    # enabled explicitly by action training/inference launchers.
    action_conditioning_enabled=False,
    action_conditioning_hidden_dim=None,
    action_conditioning_actions_per_latent=4,
    action_conditioning_clip=10.0,
    action_conditioning_scale=0.01,
    # Full-video DINO features are teacher-only by default.  Set True only to
    # reproduce legacy experiments that exposed future teacher frames as input.
    sam3d_teacher_tokens_as_condition=False,
)

mini_net = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
mini_net.model_channels = 1024
mini_net.num_heads = 8
mini_net.num_blocks = 2
mini_net.rope_t_extrapolation_ratio = 1.0


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B", node=WAN2PT1_1PT3B)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B", node=WAN2PT1_14B)
    cs.store(group="net", package="model.config.net", name="mini_net", node=mini_net)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B_sam3d", node=COSMOS_V1_2B_SAM3D_NET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_7B", node=COSMOS_V1_7B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_14B", node=COSMOS_V1_14B_NET_MININET)
