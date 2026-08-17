#!/bin/bash
#SBATCH --job-name=sam3d-sidecar
#SBATCH --partition=GPUA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/share/home/2301213237/cosmos-sam3d-work/logs/sidecar-smoke-%j.out
#SBATCH --error=/gpfs/share/home/2301213237/cosmos-sam3d-work/logs/sidecar-smoke-%j.err

set -euo pipefail

ROOT=/gpfs/share/home/2301213237/cosmos-sam3d-work
ENV=${ROOT}/envs/sam3d-objects-offline
DATA=${ROOT}/smoke-input/dataset
MASK_CACHE=${ROOT}/smoke-input/mask-cache
OUTPUT=${ROOT}/smoke-output/native-sidecars
CONFIG=${ROOT}/weights/sam3d-objects/checkpoints/pipeline.stage1.offline.yaml

[[ -f "${ENV}/.sam3d_runtime_ready" ]]
mkdir -p "${OUTPUT}" "${ROOT}/logs" "${ROOT}/tmp"
module load gcc/12.3.0 cuda/12.6
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export PYTHONPATH=${ROOT}/src/sam-3d-objects-official:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_HOME=${ROOT}/weights/torch
export SAM3D_DINOV2_REPO=${ROOT}/src/dinov2-official
export LIDRA_SKIP_INIT=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${ENV}/bin/python" "${ROOT}/scripts/prepare_sam3d_object_sidecars.py" \
  --dataset-root "${DATA}" \
  --manifest "${DATA}/manifest.jsonl" \
  --mask-cache-root "${MASK_CACHE}" \
  --output-root "${OUTPUT}" \
  --sam3d-repo "${ROOT}/src/sam-3d-objects-official" \
  --pipeline-config "${CONFIG}" \
  --limit 1 \
  --max-objects 1 \
  --overwrite

"${ENV}/bin/python" - <<'PY'
from pathlib import Path
import torch

root = Path("/gpfs/share/home/2301213237/cosmos-sam3d-work/smoke-output/native-sidecars")
files = list(root.rglob("sam3d_objects.pt"))
assert len(files) == 1, files
value = torch.load(files[0], map_location="cpu", weights_only=True)
shape = value["sam3d_shape_latents"]
geometry = value["sam3d_geometry"]
pose = value["sam3d_object_pose"]
assert shape.ndim == 3 and shape.shape[0] == 1 and shape.shape[-1] == 8
assert geometry.shape == (8, 120, 160)
assert pose.shape == (1, 10)
assert torch.isfinite(shape).all() and shape.abs().sum() > 0
assert torch.isfinite(geometry).all() and geometry.abs().sum() > 0
print(
    "SAM3D_NATIVE_SIDECAR_OK",
    files[0], tuple(shape.shape), tuple(geometry.shape), tuple(pose.shape),
    float(shape.abs().mean()), float(geometry.abs().mean()),
)
PY
echo "end=$(date -Is)"
