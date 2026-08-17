from pathlib import Path

import torch

from merge_sam3d_object_sidecars import normalized_native_conditions


sidecar = {
    "sam3d_shape_latents": torch.randn(1, 4096, 8),
    "sam3d_object_pose": torch.randn(1, 10),
    "sam3d_geometry": torch.randn(8, 120, 160),
    "object_count": torch.tensor(1),
}
shape, pose, geometry, count = normalized_native_conditions(sidecar, 8, 256, Path("synthetic.pt"))
assert shape.shape == (8, 256, 8)
assert pose.shape == (8, 10)
assert geometry.shape == (8, 120, 160)
assert count == 1
print("SAM3D_MERGER_UNIT_OK", tuple(shape.shape), tuple(pose.shape), tuple(geometry.shape), count)
