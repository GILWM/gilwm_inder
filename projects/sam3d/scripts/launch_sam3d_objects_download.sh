#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
image="${COSMOS_MUSA_IMAGE:-10.200.53.208/ci/nanhu-computing-framework:v2.1.6-rc1}"
env_bin="${workspace}/envs/cosmos-musa/bin"

exec docker run --rm --name cosmos-sam3d-objects-download --network host \
  -v /datassd:/datassd \
  -e "PATH=${env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e "HTTP_PROXY=http://127.0.0.1:17890" \
  -e "HTTPS_PROXY=http://127.0.0.1:17890" \
  -e "HF_HOME=${workspace}/weights/huggingface" \
  -e "HF_HUB_DISABLE_XET=1" \
  "${image}" bash "${workspace}/scripts/download_sam3d_objects.sh"
