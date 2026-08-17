#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
image="${COSMOS_MUSA_IMAGE:-10.200.53.208/ci/nanhu-computing-framework:v2.1.6-rc1}"
python="${workspace}/envs/cosmos-musa/bin/python"
env_bin="${workspace}/envs/cosmos-musa/bin"
pythonpath="${workspace}/third_party/sam3-musa:${workspace}/third_party/sam-3d-objects-official:${workspace}/third_party/dinov2-official:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages"

test -x "${python}"
mkdir -p "${workspace}/weights/torch"

exec docker run --rm --privileged --network host --ipc host \
  -v /datassd:/datassd \
  -v /datahdd:/datahdd \
  -w "${workspace}" \
  -e "PYTHONPATH=${pythonpath}" \
  -e "PATH=${env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e "TORCH_HOME=${workspace}/weights/torch" \
  -e "HTTP_PROXY=http://127.0.0.1:17890" \
  -e "HTTPS_PROXY=http://127.0.0.1:17890" \
  -e "NO_PROXY=127.0.0.1,localhost" \
  -e "KMP_DUPLICATE_LIB_OK=TRUE" \
  -e "KMP_USE_SHM=0" \
  -e "OMP_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "SAM3_CHECKPOINT=${workspace}/weights/sam3/sam3.pt" \
  -e "MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}" \
  "${image}" "${python}" "$@"
