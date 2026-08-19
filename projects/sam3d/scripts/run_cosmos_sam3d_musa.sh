#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:?set COSMOS_SAM3D_WORKSPACE}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project="${COSMOS_PROJECT:-$(git -C "${script_dir}" rev-parse --show-toplevel)}"
image="${COSMOS_MUSA_IMAGE:?set COSMOS_MUSA_IMAGE to an accessible MUSA runtime image}"
python="${workspace}/envs/cosmos-musa/bin/python"
env_bin="${workspace}/envs/cosmos-musa/bin"
checkpoint_root="${workspace}/weights/cosmos"
model_root="${checkpoint_root}/models"
vae_path="${model_root}/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth"
torch_home="${workspace}/weights/torch"
pythonpath="${project}:${project}/packages/cosmos-oss:${project}/packages/cosmos-cuda:${workspace}/third_party/sam3-musa:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages"

mccl_env=()
for key in \
  MCCL_IB_GID_INDEX \
  MCCL_IB_HCA \
  MCCL_SOCKET_IFNAME \
  MCCL_DEBUG \
  MCCL_DEBUG_SUBSYS \
  MCCL_PROTOS \
  MCCL_ALGOS \
  MCCL_BUFFSIZE \
  MCCL_MIN_NRINGS \
  MCCL_MAX_NRINGS; do
  if [[ -n "${!key:-}" ]]; then
    mccl_env+=(-e "${key}=${!key}")
  fi
done

for required in "${python}" "${vae_path}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required independent runtime artifact is missing: ${required}" >&2
    exit 1
  fi
done

mount_args=(-v "${workspace}:${workspace}")
for bind_path in \
  "${SAM3D_DATASET_ROOT:-}" \
  "${SAM3D_CACHE_ROOT:-}" \
  "${SAM3D_OBJECTS_ROOT:-}" \
  "${SAM3D_BASE_CHECKPOINT:-}" \
  "${SAM3D_COSMOS_OVERLAY_CHECKPOINT:-}"; do
  if [[ -n "${bind_path}" && -e "${bind_path}" && "${bind_path}" != "${workspace}"/* ]]; then
    mount_args+=(-v "${bind_path}:${bind_path}")
  fi
done

# Filtered dataset views can contain symlinks whose targets live outside the
# manifest directory. Mount each requested root so those links also resolve
# inside the container.
IFS=: read -r -a extra_bind_paths <<< "${COSMOS_EXTRA_BIND_PATHS:-}"
for bind_path in "${extra_bind_paths[@]}"; do
  if [[ -n "${bind_path}" && -e "${bind_path}" && "${bind_path}" != "${workspace}"/* ]]; then
    mount_args+=(-v "${bind_path}:${bind_path}")
  fi
done

exec docker run --rm --privileged --network host --ipc host \
  "${mount_args[@]}" \
  -w "${project}" \
  -e "PYTHONPATH=${pythonpath}" \
  -e "PATH=${env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e "HF_HOME=${checkpoint_root}/huggingface" \
  -e "TORCH_HOME=${torch_home}" \
  -e "HF_HUB_OFFLINE=1" \
  -e "TRANSFORMERS_OFFLINE=1" \
  -e "COSMOS_CHECKPOINT_DIR=${checkpoint_root}" \
  -e "COSMOS_LOCAL_MODEL_ROOT=${model_root}" \
  -e "COSMOS_WAN2PT1_VAE_PATH=${vae_path}" \
  -e "WAN_VAE_PATH=${vae_path}" \
  -e "SAM3_CHECKPOINT=${workspace}/weights/sam3/sam3.pt" \
  -e "SAM3D_COSMOS_OVERLAY_CHECKPOINT=${SAM3D_COSMOS_OVERLAY_CHECKPOINT:-}" \
  -e "IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-${workspace}/training-outputs}" \
  -e "WANDB_MODE=${WANDB_MODE:-disabled}" \
  -e "PYTORCH_MUSA_ALLOC_CONF=${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}" \
  -e "TORCH_MUSA_FSDP2_ENABLE_CE_COMM=${TORCH_MUSA_FSDP2_ENABLE_CE_COMM:-0}" \
  -e "TORCH_MUSA_FSDP2_OVERLAP_LEVEL=${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" \
  -e "KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}" \
  -e "MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "${mccl_env[@]}" \
  "${image}" "${python}" "$@"
