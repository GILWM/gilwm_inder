import torch
import numpy as np

from cosmos_predict2._src.predict2.datasets.local_datasets.sam3d_video_dataset import SAM3DVideoDataset
from cosmos_predict2._src.predict2.sam3d.video_mask_supervision import TemporalVideoMaskHead


def test_temporal_video_mask_head_supervises_masks_and_stable_track_motion():
    torch.manual_seed(11)
    head = TemporalVideoMaskHead(model_dim=8, hidden_dim=12, max_instances=2)
    student = torch.randn(2, 3 * 2 * 3, 8, requires_grad=True)
    target = torch.zeros(2, 3, 2, 2, 3)
    # Stable track zero moves one cell to the right at every latent time.
    target[:, 0, 0, 0, 0] = 1
    target[:, 1, 0, 0, 1] = 1
    target[:, 2, 0, 0, 2] = 1
    # Track one is static and valid for only one sample.
    target[0, :, 1, 1, 1] = 1
    valid = torch.ones(2, 3, 2, dtype=torch.bool)
    valid[1, :, 1] = False

    losses = head(student, target, valid, latent_frames=3, spatial_shape=(2, 3))
    assert losses.mask.isfinite() and losses.mask.item() > 0
    assert losses.tracking.isfinite() and losses.tracking.item() >= 0
    (losses.mask + losses.tracking).backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_frame_zero_is_not_a_direct_reconstruction_target():
    head = TemporalVideoMaskHead(model_dim=4, hidden_dim=4, max_instances=1)
    student = torch.randn(1, 2 * 2 * 2, 4, requires_grad=True)
    target = torch.zeros(1, 2, 1, 2, 2)
    target[:, 0, 0, 0, 0] = 1
    valid = torch.tensor([[[True], [False]]])
    losses = head(student, target, valid, latent_frames=2, spatial_shape=(2, 2))
    assert losses.mask.item() == 0.0
    assert losses.tracking.item() == 0.0


def test_dataset_selects_exact_4x_vae_source_frames(tmp_path):
    path = tmp_path / "sam3_video_teacher.pt"
    masks = torch.zeros(10, 2, 2, 3, dtype=torch.uint8)
    for frame in range(10):
        masks[frame, 0].fill_(frame)
    torch.save(
        {
            "source_frame_indices": torch.arange(10),
            "sam3_video_masks_T_K_H_W": masks,
            "sam3_video_valid_T_K": torch.ones(10, 2, dtype=torch.bool),
        },
        path,
    )
    dataset = SAM3DVideoDataset.__new__(SAM3DVideoDataset)
    dataset.sam3_video_teacher_paths = [str(path)]
    dataset.sam3_video_teacher_required = True
    dataset.sam3_video_mask_instances = 2
    dataset.sam3_video_mask_size = (2, 3)
    dataset.sam3_video_frames_per_latent = 4
    dataset.video_paths = ["sample.mp4"]
    loaded = dataset._load_sam3_video_teacher(0, np.arange(9, dtype=np.int64))
    assert loaded is not None
    selected = loaded["sam3_video_masks_B_T_K_H_W"][:, 0, 0, 0]
    torch.testing.assert_close(selected, torch.tensor([0.0, 4.0, 8.0]))
