from types import SimpleNamespace

import torch

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.sam3d_video2world_model import (
    SAM3DVideo2WorldModelRectifiedFlow,
    SAM3DVideo2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.sam3d.action_alignment import TemporalActionAlignmentHead
from cosmos_predict2._src.predict2.sam3d.conditioner import SAM3DVideo2WorldCondition


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
    ):
        try:
            SAM3DVideo2WorldModelRectifiedFlowConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid action config was accepted: {kwargs}")


if __name__ == "__main__":
    test_denoise_collects_action_feature_and_populates_losses()
    test_action_config_rejects_invalid_weights_and_layer()
    print("SAM3D action model wiring test passed")
