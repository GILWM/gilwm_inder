#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
source_workspace="${COSMOS_SOURCE_WORKSPACE:-/datassd/morka/cosmos-work}"
source_env="${COSMOS_SOURCE_ENV:-/datassd/morka/conda-envs/cosmos-musa}"
target_env="${workspace}/envs/cosmos-musa"
source_model="${source_workspace}/checkpoints/models/nvidia/Cosmos-Predict2.5-2B"
source_reason="${source_workspace}/checkpoints/models/nvidia/Cosmos-Reason1-7B"
target_checkpoint_root="${workspace}/weights/cosmos"
staging_checkpoint_root="${workspace}/weights/.cosmos-copying"

mkdir -p "${workspace}/envs" "${workspace}/weights" "${workspace}/configs"

assert_under_workspace() {
  local resolved
  resolved="$(realpath -m "$1")"
  case "${resolved}" in
    "${workspace}"/*) ;;
    *) echo "Unsafe path outside workspace: ${resolved}" >&2; exit 1 ;;
  esac
}
assert_under_workspace "${target_env}.copying"
assert_under_workspace "${staging_checkpoint_root}"

if [[ ! -e "${target_env}/bin/python" ]]; then
  rm -rf "${target_env}.copying"
  cp -a "${source_env}" "${target_env}.copying"
  mv "${target_env}.copying" "${target_env}"
fi

if [[ -L "${target_checkpoint_root}" ]]; then
  rm -rf "${staging_checkpoint_root}"
  mkdir -p "${staging_checkpoint_root}/models/nvidia" "${staging_checkpoint_root}/huggingface/hub"
  cp -a "${source_model}" "${staging_checkpoint_root}/models/nvidia/"
  rm "${target_checkpoint_root}"
  mv "${staging_checkpoint_root}" "${target_checkpoint_root}"
elif [[ ! -d "${target_checkpoint_root}/models/nvidia/Cosmos-Predict2.5-2B" ]]; then
  echo "Refusing to replace unexpected target: ${target_checkpoint_root}" >&2
  exit 1
fi

if [[ ! -d "${target_checkpoint_root}/models/nvidia/Cosmos-Reason1-7B" ]]; then
  cp -a "${source_reason}" "${target_checkpoint_root}/models/nvidia/"
fi

find "${target_env}" -type f -printf '%s %P\n' | sort -k2 >"${workspace}/configs/cosmos-musa-env.files"
find "${target_checkpoint_root}/models/nvidia/Cosmos-Predict2.5-2B" -type f -printf '%s %P\n' | sort -k2 \
  >"${workspace}/configs/cosmos-predict2.5-2b-weights.files"
find "${target_checkpoint_root}/models/nvidia/Cosmos-Reason1-7B" -type f -printf '%s %P\n' | sort -k2 \
  >"${workspace}/configs/cosmos-reason1-7b-weights.files"

"${target_env}/bin/python" -m pip freeze >"${workspace}/configs/cosmos-musa-pip-freeze.txt"
docker image inspect 10.200.53.208/ci/nanhu-computing-framework:v2.1.6-rc1 --format '{{.Id}}' \
  >"${workspace}/configs/docker-image.digest"

du -sh "${target_env}" \
  "${target_checkpoint_root}/models/nvidia/Cosmos-Predict2.5-2B" \
  "${target_checkpoint_root}/models/nvidia/Cosmos-Reason1-7B"
touch "${workspace}/configs/runtime-copy.complete"
echo RUNTIME_COPY_COMPLETE
