from __future__ import annotations

import torch

from cosmos_predict2._src.predict2.networks.sam3d_conditioned_dit import (
    ActionChunkConditioner,
    LocalTemporalActionMixer,
)


def test_temporal_mixer_is_identity_at_initialization():
    mixer = LocalTemporalActionMixer(action_dim=3, radius=2, hidden_dim=8)
    actions = torch.randn(2, 9, 3)
    torch.testing.assert_close(mixer(actions), actions)


def test_temporal_mixer_can_select_future_and_use_deltas():
    mixer = LocalTemporalActionMixer(action_dim=1, radius=1, hidden_dim=4)
    with torch.no_grad():
        mixer.mix_gate.fill_(10.0)
        mixer.delta_gate.zero_()
        mixer.relative_logits.copy_(torch.tensor([-20.0, -20.0, 20.0]))
    actions = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    mixed = mixer(actions)
    torch.testing.assert_close(mixed[:, 1:4], actions[:, 2:5], atol=1.0e-4, rtol=0.0)


def test_safe_conditioner_starts_as_exact_no_action_and_receives_gradient():
    conditioner = ActionChunkConditioner(
        action_dim=2,
        model_dim=8,
        actions_per_latent=4,
        hidden_dim=16,
        temporal_context_radius=2,
        temporal_context_hidden_dim=8,
        output_norm="tanh",
        zero_init_output=True,
        conditioning_dropout=0.0,
        residual_scale=0.01,
    )
    actions = torch.randn(2, 9, 2)
    output, adaln = conditioner(actions, latent_frames=3)
    assert adaln is None
    assert output.shape == (2, 3, 8)
    torch.testing.assert_close(output, torch.zeros_like(output))

    output.sum().backward()
    final_weight = conditioner.timestep_embedder[2].weight
    assert final_weight.grad is not None
    assert torch.count_nonzero(final_weight.grad).item() > 0


def test_conditioner_accepts_symmetric_action_halo_for_ar_chunks():
    conditioner = ActionChunkConditioner(
        action_dim=2,
        model_dim=8,
        actions_per_latent=4,
        hidden_dim=16,
        temporal_context_radius=2,
        temporal_context_hidden_dim=8,
        output_norm="tanh",
        zero_init_output=False,
    )
    # Three latent frames normally require 9 actions. AR inference may supply
    # two real neighboring actions on each side, which are mixed first and
    # cropped back to the exact 9-frame center timeline.
    output, _ = conditioner(torch.randn(2, 13, 2), latent_frames=3)
    assert output.shape == (2, 3, 8)


def test_halo_mix_then_crop_preserves_global_future_indices():
    mixer = LocalTemporalActionMixer(action_dim=1, radius=2, hidden_dim=4)
    with torch.no_grad():
        mixer.mix_gate.fill_(10.0)
        mixer.relative_logits.copy_(torch.tensor([-20.0, -20.0, -20.0, -20.0, 20.0]))
    # The center chunk is global frames 2..10. Selecting offset +2 before
    # cropping must therefore yield global actions 4..12, including the true
    # right-side AR halo rather than a repeated chunk-edge action.
    global_actions = torch.arange(13, dtype=torch.float32).reshape(1, 13, 1)
    center = mixer(global_actions)[:, 2:11]
    torch.testing.assert_close(center, global_actions[:, 4:13], atol=1.0e-4, rtol=0.0)


def test_action_dropout_is_training_only():
    conditioner = ActionChunkConditioner(
        action_dim=2,
        model_dim=8,
        actions_per_latent=4,
        hidden_dim=16,
        output_norm="tanh",
        zero_init_output=False,
        conditioning_dropout=1.0,
    )
    actions = torch.randn(2, 9, 2)
    conditioner.train()
    train_output, _ = conditioner(actions, latent_frames=3)
    torch.testing.assert_close(train_output, torch.zeros_like(train_output))

    conditioner.eval()
    eval_output, _ = conditioner(actions, latent_frames=3)
    assert torch.count_nonzero(eval_output).item() > 0


def test_legacy_configuration_retains_layernorm_chunk_path():
    conditioner = ActionChunkConditioner(
        action_dim=2,
        model_dim=8,
        actions_per_latent=4,
        hidden_dim=16,
        temporal_context_radius=0,
        output_norm="layernorm",
        zero_init_output=False,
        conditioning_dropout=0.0,
        residual_scale=0.01,
    )
    assert conditioner.temporal_mixer is None
    output, _ = conditioner(torch.randn(2, 9, 2), latent_frames=3)
    assert output.shape == (2, 3, 8)
    torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
    assert torch.count_nonzero(output[:, 1:]).item() > 0


def test_temporal_conditioner_state_dict_roundtrip():
    kwargs = dict(
        action_dim=2,
        model_dim=8,
        actions_per_latent=4,
        hidden_dim=16,
        temporal_context_radius=2,
        temporal_context_hidden_dim=8,
        output_norm="tanh",
        zero_init_output=True,
        conditioning_dropout=0.1,
    )
    source = ActionChunkConditioner(**kwargs).eval()
    with torch.no_grad():
        source.timestep_embedder[2].weight.normal_(std=0.01)
        source.temporal_mixer.mix_gate.fill_(0.2)
        source.temporal_mixer.relative_logits.copy_(torch.tensor([-1.0, 0.0, 2.0, 0.0, -1.0]))
    restored = ActionChunkConditioner(**kwargs).eval()
    restored.load_state_dict(source.state_dict(), strict=True)

    actions = torch.randn(2, 9, 2)
    expected, _ = source(actions, latent_frames=3)
    actual, _ = restored(actions, latent_frames=3)
    torch.testing.assert_close(actual, expected)
