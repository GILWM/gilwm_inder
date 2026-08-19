import torch

from cosmos_predict2._src.predict2.sam3d.action_alignment import TemporalActionAlignmentHead


def test_time_aligned_action_losses_are_finite_and_mask_invalid_samples():
    torch.manual_seed(17)
    head = TemporalActionAlignmentHead(model_dim=8, action_dim=14, hidden_dim=4)
    video = torch.randn(2, 3 * 5, 8, requires_grad=True)
    actions = torch.randn(2, 9, 14)
    losses = head(video, actions, latent_frames=3, valid_B=torch.tensor([True, False]))
    assert losses.prediction.isfinite()
    assert losses.alignment.isfinite()
    (losses.prediction + losses.alignment).backward()
    assert torch.count_nonzero(video.grad[0])
    assert torch.count_nonzero(video.grad[1]) == 0
    assert head.action_encoder(actions[:, :1]).shape == (2, 1, 8)


def test_latent_indices_preserve_first_and_last_action_frames():
    indices = TemporalActionAlignmentHead.latent_action_indices(93, 24, torch.device("cpu"))
    torch.testing.assert_close(indices, torch.arange(0, 93, 4))


def test_all_invalid_actions_return_differentiable_zero():
    head = TemporalActionAlignmentHead(model_dim=8, action_dim=14, hidden_dim=4)
    video = torch.randn(2, 15, 8, requires_grad=True)
    actions = torch.randn(2, 9, 14)
    losses = head(video, actions, latent_frames=3, valid_B=torch.tensor([False, False]))
    assert losses.prediction.item() == losses.alignment.item() == 0.0
    losses.prediction.backward()
    assert video.grad is not None


if __name__ == "__main__":
    test_time_aligned_action_losses_are_finite_and_mask_invalid_samples()
    test_latent_indices_preserve_first_and_last_action_frames()
    test_all_invalid_actions_return_differentiable_zero()
    print("Temporal action alignment tests passed")
