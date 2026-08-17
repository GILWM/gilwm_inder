import importlib.util

import torch
import transformers


print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"transformers_sam3={hasattr(transformers, 'Sam3Model')}")
print(f"torchvision_present={importlib.util.find_spec('torchvision') is not None}")
print(f"pytorch3d_present={importlib.util.find_spec('pytorch3d') is not None}")
print(f"kaolin_present={importlib.util.find_spec('kaolin') is not None}")
print(f"nvdiffrast_present={importlib.util.find_spec('nvdiffrast') is not None}")
for module in (
    "torch_musa",
    "timm",
    "ftfy",
    "regex",
    "iopath",
    "decord",
    "cv2",
    "PIL",
    "huggingface_hub",
):
    print(f"{module}_present={importlib.util.find_spec(module) is not None}")
print(f"musa_available={getattr(torch, 'musa', None) is not None and torch.musa.is_available()}")
