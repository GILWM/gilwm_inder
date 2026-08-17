from types import SimpleNamespace

import torch

from cosmos_predict2._src.predict2.models.sam3d_video2world_model import (
    SAM3DVideo2WorldModelRectifiedFlow,
)
from cosmos_predict2._src.predict2.networks.sam3d_conditioned_dit import ConditionProjector


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_spatial = 2
        self.sam3d_repa_projector = ConditionProjector(12, 6)


def make_model() -> SAM3DVideo2WorldModelRectifiedFlow:
    model = SAM3DVideo2WorldModelRectifiedFlow.__new__(SAM3DVideo2WorldModelRectifiedFlow)
    torch.nn.Module.__init__(model)
    model.net = ToyNet()
    model.config = SimpleNamespace(
        sam3d_repa_frames=3,
        sam3d_repa_grid_size=4,
        sam3d_repa_temporal_weight=0.25,
    )
    return model


def test_projected_repa_is_finite_and_ignores_zero_dropout_samples():
    torch.manual_seed(11)
    model = make_model()
    # latent H,W=8,10 and patch_spatial=2 -> student grid 4x5.
    student = torch.randn(2, 3 * 4 * 5, 12, requires_grad=True)
    teacher = torch.randn(2, 3, 16, 6)
    teacher[1].zero_()  # Mimics per-sample conditioner dropout.

    total, spatial, temporal = model._projected_cosine_alignment_loss(
        student,
        teacher,
        latent_frames=3,
        latent_height=8,
        latent_width=10,
    )
    assert total.isfinite() and spatial.isfinite() and temporal.isfinite()
    torch.testing.assert_close(total, spatial + 0.25 * temporal)
    total.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[0])
    assert torch.count_nonzero(student.grad[1]) == 0


def test_all_zero_teachers_return_differentiable_zero():
    model = make_model()
    student = torch.randn(2, 3 * 4 * 5, 12, requires_grad=True)
    teacher = torch.zeros(2, 3, 16, 6)
    total, spatial, temporal = model._projected_cosine_alignment_loss(
        student,
        teacher,
        latent_frames=3,
        latent_height=8,
        latent_width=10,
    )
    assert total.item() == spatial.item() == temporal.item() == 0.0
    total.backward()
    assert student.grad is not None


def test_temporal_delta_penalizes_a_static_student():
    model = make_model()
    # All student frames are identical, while the teacher changes over time.
    frame = torch.randn(1, 4 * 5, 12)
    student = frame.repeat(1, 3, 1).requires_grad_()
    teacher = torch.randn(1, 3, 16, 6)
    _, _, temporal = model._projected_cosine_alignment_loss(
        student,
        teacher,
        latent_frames=3,
        latent_height=8,
        latent_width=10,
    )
    assert temporal.isfinite()
    assert temporal.item() > 0.5
    temporal.backward()
    assert torch.count_nonzero(student.grad)


if __name__ == "__main__":
    test_projected_repa_is_finite_and_ignores_zero_dropout_samples()
    test_all_zero_teachers_return_differentiable_zero()
    test_temporal_delta_penalizes_a_static_student()
    print("projected REPA tests passed")
