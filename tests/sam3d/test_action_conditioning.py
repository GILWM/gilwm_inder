import torch

from cosmos_predict2._src.predict2.networks.sam3d_conditioned_dit import ActionChunkConditioner


def test_action_chunk_conditioner_can_be_zero_scaled_and_output_preserving():
    module = ActionChunkConditioner(
        action_dim=14, model_dim=8, actions_per_latent=4, hidden_dim=16, residual_scale=0.0
    )
    actions = torch.randn(2, 93, 14)
    timestep, adaln = module(actions, latent_frames=24)
    assert timestep.shape == (2, 24, 8)
    assert adaln is None
    assert torch.count_nonzero(timestep) == 0
    assert torch.count_nonzero(module.timestep_embedder[2].weight) > 0
    assert module.residual_scale == 0.0


def test_action_chunks_use_exact_future_frame_groups():
    module = ActionChunkConditioner(
        action_dim=2,
        model_dim=4,
        actions_per_latent=4,
        hidden_dim=8,
        action_clip=None,
    )
    actions = torch.arange(9 * 2, dtype=torch.float32).reshape(1, 9, 2)
    captured = []
    hook = module.timestep_embedder[0].register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].clone()))
    try:
        module(actions, latent_frames=3)
    finally:
        hook.remove()
    expected = torch.stack((actions[0, 1:5].flatten(), actions[0, 5:9].flatten())).unsqueeze(0)
    torch.testing.assert_close(captured[0], expected)


def test_action_chunk_conditioner_rejects_off_by_one_trajectories():
    module = ActionChunkConditioner(action_dim=14, model_dim=8, actions_per_latent=4, hidden_dim=16)
    for length in (92, 94):
        try:
            module(torch.randn(1, length, 14), latent_frames=24)
        except ValueError as error:
            assert "alignment mismatch" in str(error)
        else:
            raise AssertionError(f"Accepted misaligned action length {length}")


def test_action_conditioning_is_trainable_and_becomes_action_sensitive():
    torch.manual_seed(41)
    module = ActionChunkConditioner(
        action_dim=2, model_dim=4, actions_per_latent=4, hidden_dim=16, residual_scale=1.0
    )
    actions_a = torch.zeros(1, 9, 2)
    actions_b = torch.ones(1, 9, 2)
    # Layer-normalized residuals are deliberately zero mean, so use a
    # representable non-constant target when checking optimizer progress.
    target = torch.tensor([0.75, -0.75, 0.5, -0.5]).reshape(1, 1, 4).expand(1, 3, 4)
    optimizer = torch.optim.AdamW(module.parameters(), lr=2.0e-2, weight_decay=0.0)
    history = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = module(actions_b, latent_frames=3)
        loss = torch.nn.functional.mse_loss(prediction[:, 1:], target[:, 1:])
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    assert min(history[-10:]) < 0.25 * history[0]
    prediction_a, _ = module(actions_a, latent_frames=3)
    prediction_b, _ = module(actions_b, latent_frames=3)
    assert not torch.allclose(prediction_a[:, 1:], prediction_b[:, 1:])


def test_invalid_or_observed_frames_receive_no_action_conditioning():
    module = ActionChunkConditioner(
        action_dim=2, model_dim=4, actions_per_latent=4, hidden_dim=8, residual_scale=1.0
    )
    actions = torch.randn(2, 9, 2)
    condition_mask = torch.zeros(2, 1, 3, 1, 1)
    condition_mask[:, :, 0] = 1
    timestep, adaln = module(
        actions,
        latent_frames=3,
        action_valid_B=torch.tensor([True, False]),
        condition_video_input_mask_B_C_T_H_W=condition_mask,
    )
    assert torch.count_nonzero(timestep[:, 0]) == 0
    assert adaln is None
    assert torch.count_nonzero(timestep[1]) == 0
    assert torch.count_nonzero(timestep[0, 1:]) > 0


def test_extreme_normalized_actions_are_clipped_before_embedding():
    module = ActionChunkConditioner(
        action_dim=2,
        model_dim=4,
        actions_per_latent=4,
        hidden_dim=8,
        action_clip=10.0,
    )
    captured = []
    hook = module.timestep_embedder[0].register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    actions = torch.tensor([[[0.0, 0.0]] + [[100.0, -100.0]] * 8])
    try:
        module(actions, latent_frames=3)
    finally:
        hook.remove()
    assert captured[0].max().item() == 10.0
    assert captured[0].min().item() == -10.0
