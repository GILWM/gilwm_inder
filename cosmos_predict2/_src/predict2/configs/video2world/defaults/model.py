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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.models.text2world_wan2pt1_model import Text2WorldModelWan2pt1Config
from cosmos_predict2._src.predict2.models.video2world_model import Video2WorldConfig, Video2WorldModel
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.models.video2world_wan2pt1_model import I2VWan2pt1Model
from cosmos_predict2._src.predict2.models.sam3d_video2world_model import (
    SAM3DVideo2WorldModelRectifiedFlow,
    SAM3DVideo2WorldModelRectifiedFlowConfig,
)

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(Video2WorldModel)(
        config=Video2WorldConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Video2WorldModel)(
        config=Video2WorldConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


FSDP_WAN2PT1_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(I2VWan2pt1Model)(
        config=Text2WorldModelWan2pt1Config(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)

FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Video2WorldModelRectifiedFlow)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)

FSDP_SAM3D_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(SAM3DVideo2WorldModelRectifiedFlow)(
        config=SAM3DVideo2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
            sam3d_repa_weight=0.5,
            sam3d_repa_layer=13,
            sam3d_repa_mode="relation",
            sam3d_repa_grid_size=16,
            sam3d_repa_temporal_weight=0.25,
            action_loss_weight=0.0,
            action_alignment_weight=0.1,
            action_feature_layer=7,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1", node=FSDP_WAN2PT1_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_rectified_flow", node=FSDP_RECTIFIED_FLOW_CONFIG)
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_sam3d_rectified_flow",
        node=FSDP_SAM3D_RECTIFIED_FLOW_CONFIG,
    )
