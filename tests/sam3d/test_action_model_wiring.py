from types import SimpleNamespace

import torch

from cosmos_predict2.experiments.sam3d_world_model import (
    predict2_video2world_inference_2b_sam3d_action,
)

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.predict2.configs.video2world.config import make_config
from cosmos_predict2._src.predict2.models.sam3d_video2world_model import (
    SAM3DVideo2WorldModelRectifiedFlow,
    SAM3DVideo2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.sam3d.action_alignment import (
    CosmosActionExpertHead,
    TemporalActionAlignmentHead,
)
from cosmos_predict2._src.predict2.sam3d.conditioner import SAM3DVideo2WorldCondition


def test_action_inference_experiment_instantiates_training_architecture():
    net = predict2_video2world_inference_2b_sam3d_action["model"]["config"]["net"]
    assert net == {
        "action_conditioning_enabled": True,
        "action_conditioning_hidden_dim": 8192,
        "action_conditioning_actions_per_latent": 4,
        "action_conditioning_clip": 10.0,
        "action_conditioning_scale": 0.01,
    }
    composed = override(
        make_config(),
        ["--", "experiment=predict2_video2world_inference_2b_sam3d_action"],
    )
    assert composed.model.config.net.action_conditioning_enabled is True
    assert composed.model.config.net.action_conditioning_hidden_dim == 8192
    assert composed.model.config.net.action_conditioning_scale == 0.01


class ToyActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_supervision_head = TemporalActionAlignmentHead(8, 14, 4)
        self.requested_features = None

    def forward(self, x_B_C_T_H_W, intermediate_feature_ids=None, **kwargs):
        del kwargs
        self.requested_features = intermediate_feature_ids
        batch, _, frames, _, _ = x_B_C_T_H_W.shape
        # Two spatial tokens per latent frame, derived from x so gradients can
        # flow through the same path as a real intermediate DiT feature.
        base = x_B_C_T_H_W.mean(dim=(1, 3, 4), keepdim=False)
        feature = base[:, :, None, None].expand(batch, frames, 2, 8).reshape(batch, frames * 2, 8)
        return torch.zeros_like(x_B_C_T_H_W), [feature]


class ToyActionExpertNet(torch.nn.Module):
    patch_spatial = 1

    def __init__(self):
        super().__init__()
        self.action_supervision_head = CosmosActionExpertHead(
            model_dim=8,
            action_dim=14,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            ffn_multiplier=2,
            pool_grid=1,
            time_embedding_dim=8,
        )
        self.requested_features = None

    def forward(self, x_B_C_T_H_W, intermediate_feature_ids=None, **kwargs):
        del kwargs
        self.requested_features = intermediate_feature_ids
        batch, _, frames, height, width = x_B_C_T_H_W.shape
        output = x_B_C_T_H_W * 0.5 + 0.125
        if not intermediate_feature_ids:
            return output
        base = x_B_C_T_H_W.mean(dim=1).flatten(1, 3).unsqueeze(-1)
        feature = base.expand(batch, frames * height * width, 8)
        return output, [feature * (layer + 1) for layer in intermediate_feature_ids]


class ToyActionConditionedInferenceNet(torch.nn.Module):
    def forward(self, x_B_C_T_H_W, actions_B_T_D=None, action_valid_B=None, **kwargs):
        del kwargs, action_valid_B
        if actions_B_T_D is None:
            raise RuntimeError("actions_B_T_D is required")
        action_signal = actions_B_T_D.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return x_B_C_T_H_W * 0.5 + action_signal


def test_denoise_collects_action_feature_and_populates_losses():
    model = SAM3DVideo2WorldModelRectifiedFlow.__new__(SAM3DVideo2WorldModelRectifiedFlow)
    torch.nn.Module.__init__(model)
    model.net = ToyActionNet()
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.config = SimpleNamespace(
        conditional_frame_timestep=-1,
        denoise_replace_gt_frames=False,
        sam3d_repa_weight=0.0,
        sam3d_repa_layer=7,
        action_loss_weight=1.0,
        action_feature_layer=5,
    )
    model.train()

    state = torch.randn(2, 4, 3, 2, 2, requires_grad=True)
    condition = SAM3DVideo2WorldCondition(
        crossattn_emb=torch.zeros(2, 1, 1),
        data_type=DataType.VIDEO,
        fps=torch.ones(2),
        use_video_condition=False,
        gt_frames=torch.zeros_like(state),
        condition_video_input_mask_B_C_T_H_W=torch.zeros(2, 1, 3, 2, 2),
        actions_B_T_D=torch.randn(2, 9, 14),
        action_valid_B=torch.tensor([True, True]),
    )
    output = model.denoise(
        noise=torch.randn_like(state),
        xt_B_C_T_H_W=state,
        timesteps_B_T=torch.zeros(2, 1),
        condition=condition,
    )
    assert output.shape == state.shape
    assert model.net.requested_features == [5]
    assert model._pending_action_prediction_loss.isfinite()
    assert model._pending_action_alignment_loss.isfinite()
    (model._pending_action_prediction_loss + model._pending_action_alignment_loss).backward()
    assert state.grad is not None


def test_action_config_rejects_invalid_weights_and_layer():
    config = SAM3DVideo2WorldModelRectifiedFlowConfig(
        action_loss_weight=1.0,
        action_alignment_weight=0.1,
        action_feature_layer=7,
    )
    assert config.action_loss_weight == 1.0

    for kwargs in (
        {"action_loss_weight": -1.0},
        {"action_alignment_weight": -1.0},
        {"action_feature_layer": -1},
        {"action_feature_layers": (1, -1)},
    ):
        try:
            SAM3DVideo2WorldModelRectifiedFlowConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid action config was accepted: {kwargs}")


def test_denoise_routes_a_cosmos_feature_pyramid_to_action_expert():
    model = SAM3DVideo2WorldModelRectifiedFlow.__new__(SAM3DVideo2WorldModelRectifiedFlow)
    torch.nn.Module.__init__(model)
    model.net = ToyActionExpertNet()
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.config = SimpleNamespace(
        conditional_frame_timestep=-1,
        denoise_replace_gt_frames=False,
        sam3d_repa_weight=0.0,
        sam3d_repa_layer=7,
        action_loss_weight=1.0,
        action_feature_layer=5,
        action_feature_layers=(2, 5),
    )
    model.train()
    state = torch.randn(2, 4, 3, 2, 2, requires_grad=True)
    condition = SAM3DVideo2WorldCondition(
        crossattn_emb=torch.zeros(2, 1, 1),
        data_type=DataType.VIDEO,
        fps=torch.ones(2),
        use_video_condition=False,
        gt_frames=torch.zeros_like(state),
        condition_video_input_mask_B_C_T_H_W=torch.zeros(2, 1, 3, 2, 2),
        actions_B_T_D=torch.randn(2, 9, 14),
        action_valid_B=torch.tensor([True, True]),
    )
    model.denoise(
        noise=torch.randn_like(state),
        xt_B_C_T_H_W=state,
        timesteps_B_T=torch.tensor([[0.1], [0.9]]),
        condition=condition,
    )
    assert model.net.requested_features == [2, 5]
    assert model._pending_action_prediction_loss.isfinite()
    assert model._pending_action_alignment_loss.isfinite()
    (model._pending_action_prediction_loss + model._pending_action_alignment_loss).backward()
    assert state.grad is not None


def test_denoise_passes_actions_during_inference_and_actions_change_prediction():
    model = SAM3DVideo2WorldModelRectifiedFlow.__new__(SAM3DVideo2WorldModelRectifiedFlow)
    torch.nn.Module.__init__(model)
    model.net = ToyActionConditionedInferenceNet()
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.config = SimpleNamespace(
        conditional_frame_timestep=-1,
        denoise_replace_gt_frames=False,
        sam3d_repa_weight=0.0,
        sam3d_repa_layer=7,
        action_loss_weight=0.0,
        action_feature_layer=5,
        action_feature_layers=(2, 5),
    )
    model.eval()
    state = torch.randn(2, 4, 3, 2, 2)
    noise = torch.randn_like(state)
    common = dict(
        crossattn_emb=torch.zeros(2, 1, 1),
        data_type=DataType.VIDEO,
        fps=torch.ones(2),
        use_video_condition=False,
        gt_frames=torch.zeros_like(state),
        condition_video_input_mask_B_C_T_H_W=torch.zeros(2, 1, 3, 2, 2),
    )
    condition_a = SAM3DVideo2WorldCondition(
        **common,
        actions_B_T_D=torch.zeros(2, 9, 14),
        action_valid_B=torch.tensor([True, True]),
    )
    condition_b = SAM3DVideo2WorldCondition(
        **common,
        actions_B_T_D=torch.ones(2, 9, 14),
        action_valid_B=torch.tensor([True, True]),
    )
    with torch.no_grad():
        prediction_a = model.denoise(
            noise=noise,
            xt_B_C_T_H_W=state,
            timesteps_B_T=torch.tensor([[0.1], [0.9]]),
            condition=condition_a,
        )
        prediction_b = model.denoise(
            noise=noise,
            xt_B_C_T_H_W=state,
            timesteps_B_T=torch.tensor([[0.1], [0.9]]),
            condition=condition_b,
        )
    assert not torch.equal(prediction_a, prediction_b)


if __name__ == "__main__":
    test_denoise_collects_action_feature_and_populates_losses()
    test_action_config_rejects_invalid_weights_and_layer()
    test_denoise_routes_a_cosmos_feature_pyramid_to_action_expert()
    test_denoise_passes_actions_during_inference_and_actions_change_prediction()
    print("SAM3D action model wiring test passed")
