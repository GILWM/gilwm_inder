import torch
import torch_musa  # noqa: F401

import sam3.model_builder  # noqa: F401

print(f"SAM3_IMPORT_OK musa={torch.musa.is_available()}")
