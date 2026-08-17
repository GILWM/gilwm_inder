import torch

from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.networks.sam3d_conditioned_dit import MinimalV1LVGSam3DDiT


def test_zero_gates_preserve_cosmos_and_conditions_can_change_output():
    torch.manual_seed(7)
    device = torch.device("musa")
    model = MinimalV1LVGSam3DDiT(
        max_img_h=8,
        max_img_w=8,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        crossattn_emb_channels=32,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=64,
        concat_padding_mask=True,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=16,
        atten_backend="minimal_a2a",
        sac_config=SACConfig(),
        sam3d_token_dim=24,
        sam_mask_instances=3,
        sam_mask_meta_dim=8,
        sam_geometry_channels=8,
        sam_condition_hidden_dim=16,
        sam_condition_grid_size=2,
        sam_condition_max_tokens=8,
        sam3d_repa_projection_dim=24,
    ).to(device=device, dtype=torch.bfloat16).eval()

    x = torch.randn(1, 16, 2, 8, 8, device=device, dtype=torch.bfloat16)
    timestep = torch.full((1, 1), 500.0, device=device)
    # Production Cosmos Reason1 produces a single flattened input token which
    # the DiT first projects into crossattn_emb_channels.
    text = torch.randn(1, 1, 64, device=device, dtype=torch.bfloat16)
    frame_mask = torch.zeros(1, 1, 2, 8, 8, device=device, dtype=torch.bfloat16)
    padding_mask = torch.zeros(1, 8, 8, device=device, dtype=torch.bfloat16)
    fps = torch.tensor([16.0], device=device)
    condition = dict(
        sam3d_tokens_B_F_N_D=torch.randn(1, 2, 12, 24, device=device, dtype=torch.bfloat16),
        sam_mask_B_K_H_W=torch.rand(1, 3, 8, 8, device=device, dtype=torch.bfloat16),
        sam_mask_meta_B_K_D=torch.rand(1, 3, 8, device=device, dtype=torch.bfloat16),
        sam3d_geometry_B_C_H_W=torch.rand(1, 8, 8, 8, device=device, dtype=torch.bfloat16),
        sam3d_shape_latents_B_K_N_D=torch.rand(1, 3, 12, 8, device=device, dtype=torch.bfloat16),
        sam3d_object_pose_B_K_D=torch.rand(1, 3, 10, device=device, dtype=torch.bfloat16),
    )

    with torch.no_grad():
        baseline = model(x, timestep, text, frame_mask, fps=fps, padding_mask=padding_mask)
        gated_zero = model(x, timestep, text, frame_mask, fps=fps, padding_mask=padding_mask, **condition)
        sam_context = model._build_sam_context(output_dtype=torch.bfloat16, **condition)
    torch.testing.assert_close(baseline, gated_zero)
    assert sam_context is not None
    assert sam_context.shape == (1, 8, 32)
    assert model.sam3d_repa_projector is not None
    projected = model.sam3d_repa_projector(torch.randn(1, 5, 64, device=device))
    assert projected.shape == (1, 5, 24)

    with torch.no_grad():
        model.sam_context_block_gates.fill_(0.2)
        conditioned, features = model(
            x,
            timestep,
            text,
            frame_mask,
            fps=fps,
            padding_mask=padding_mask,
            intermediate_feature_ids=[0],
            **condition,
        )
    assert conditioned.shape == baseline.shape
    assert len(features) == 1
    assert not torch.allclose(conditioned, baseline)


if __name__ == "__main__":
    test_zero_gates_preserve_cosmos_and_conditions_can_change_output()
    print("sam3d_conditioned_dit smoke test passed")
