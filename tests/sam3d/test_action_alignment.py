import torch

from cosmos_predict2._src.predict2.sam3d.action_alignment import (
    CosmosActionExpertHead,
    TemporalActionAlignmentHead,
)


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


def test_cosmos_action_expert_reads_feature_pyramid_without_action_leakage():
    torch.manual_seed(23)
    head = CosmosActionExpertHead(
        model_dim=8,
        action_dim=14,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        ffn_multiplier=2,
        pool_grid=2,
        time_embedding_dim=8,
    )
    features = [
        torch.randn(2, 3 * 2 * 3, 8, requires_grad=True),
        torch.randn(2, 3 * 2 * 3, 8, requires_grad=True),
    ]
    actions = torch.randn(2, 9, 14)
    losses = head(
        features,
        actions,
        latent_frames=3,
        valid_B=torch.tensor([True, False]),
        spatial_shape=(2, 3),
        timesteps_B_T=torch.tensor([[0.2], [0.8]]),
    )
    assert losses.prediction.isfinite()
    assert losses.alignment.isfinite()
    (losses.prediction + losses.alignment).backward()
    for feature in features:
        assert torch.count_nonzero(feature.grad[0])
        assert torch.count_nonzero(feature.grad[1]) == 0
    # Actions are targets only; they do not require gradients or enter the
    # expert's query construction.
    assert actions.grad is None


def test_cosmos_action_expert_rejects_incorrect_spatial_shape():
    head = CosmosActionExpertHead(
        model_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_multiplier=2,
        pool_grid=1,
        time_embedding_dim=8,
    )
    try:
        head(
            torch.randn(1, 18, 8),
            torch.randn(1, 9, 14),
            latent_frames=3,
            spatial_shape=(2, 2),
        )
    except ValueError as error:
        assert "spatial_shape" in str(error)
    else:
        raise AssertionError("Incorrect spatial shape was accepted")


def test_cosmos_action_expert_loss_can_overfit_a_fixed_minibatch():
    """Guard against a finite-but-disconnected auxiliary loss."""

    torch.manual_seed(29)
    head = CosmosActionExpertHead(
        model_dim=8,
        action_dim=14,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_multiplier=2,
        pool_grid=1,
        time_embedding_dim=8,
    )
    features = [torch.randn(1, 2, 8)]
    actions = torch.randn(1, 5, 14)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2.0e-2, weight_decay=0.0)

    history = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        losses = head(
            features,
            actions,
            latent_frames=2,
            spatial_shape=(1, 1),
            timesteps_B_T=torch.tensor([[0.4]]),
        )
        objective = losses.prediction + 0.1 * losses.alignment
        objective.backward()
        optimizer.step()
        history.append(float(losses.prediction.detach()))

    assert min(history[-10:]) < 0.05 * history[0], (history[0], min(history[-10:]))


if __name__ == "__main__":
    test_time_aligned_action_losses_are_finite_and_mask_invalid_samples()
    test_latent_indices_preserve_first_and_last_action_frames()
    test_all_invalid_actions_return_differentiable_zero()
    test_cosmos_action_expert_reads_feature_pyramid_without_action_leakage()
    test_cosmos_action_expert_rejects_incorrect_spatial_shape()
    test_cosmos_action_expert_loss_can_overfit_a_fixed_minibatch()
    print("Temporal action alignment tests passed")
