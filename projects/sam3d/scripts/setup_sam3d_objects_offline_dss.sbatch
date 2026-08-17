#!/bin/bash
#SBATCH --job-name=sam3d-env
#SBATCH --partition=GPUA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/share/home/2301213237/cosmos-sam3d-work/logs/setup-offline-%j.out
#SBATCH --error=/gpfs/share/home/2301213237/cosmos-sam3d-work/logs/setup-offline-%j.err

set -euo pipefail

ROOT=/gpfs/share/home/2301213237/cosmos-sam3d-work
ENV=${ROOT}/envs/sam3d-objects-offline
PY=${ENV}/bin/python
WHEELS=${ROOT}/offline/bundle/wheels
SOURCES=${ROOT}/offline/bundle/sources
BUILD=${ROOT}/src/offline-build
STATE=${ROOT}/envs/.sam3d-offline-state
TMP=${ROOT}/tmp/setup-${SLURM_JOB_ID}

mkdir -p "${BUILD}" "${STATE}" "${TMP}" "${ROOT}/logs"
[[ -f "${ENV}/.sam3d_base_ready" && -x "${PY}" ]]

export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_HOME=${ROOT}/weights/torch
export PYTHONPATH=${ROOT}/src/sam-3d-objects-official:${PYTHONPATH:-}
export TMPDIR=${TMP}
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-7}
export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-7}
export TORCH_CUDA_ARCH_LIST=8.0
export LIDRA_SKIP_INIT=true

# The cluster exposes 12.6 as its installed compiler.  PyTorch itself remains
# the cloned, known-good 2.5.1+cu121 runtime; only custom extension compilation
# uses the newer CUDA 12.x toolkit.
# The compute-node system compiler is GCC 4.8, which PyTorch 2.5 refuses for
# C++/CUDA extensions.  Use the cluster-provided modern compiler explicitly.
module load gcc/12.3.0 cuda/12.6
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export PATH=${CUDA_HOME}/bin:${PATH}

echo "node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PY}" - <<'PY'
import torch
print("BASE", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
nvcc --version | tail -n 1
gcc --version | head -n 1

install_wheels() {
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/typing_extensions-4.16.0-py3-none-any.whl" \
    "${WHEELS}/termcolor-3.3.0-py3-none-any.whl" \
    "${WHEELS}/portalocker-4.1.0-py3-none-any.whl" \
    "${WHEELS}/yacs-0.1.8-py3-none-any.whl" \
    "${WHEELS}/pybind11-3.1.0-py3-none-any.whl" \
    "${WHEELS}/ninja-1.13.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
    "${WHEELS}/fire-0.7.1-py3-none-any.whl" \
    "${WHEELS}/lark-1.3.1-py3-none-any.whl" \
    "${WHEELS}/platformdirs-4.11.2-py3-none-any.whl" \
    "${WHEELS}/mdurl-0.1.2-py3-none-any.whl" \
    "${WHEELS}/markdown_it_py-4.2.0-py3-none-any.whl" \
    "${WHEELS}/pygments-2.20.0-py3-none-any.whl" \
    "${WHEELS}/rich-15.0.0-py3-none-any.whl" \
    "${WHEELS}/docstring_parser-0.18.0-py3-none-any.whl" \
    "${WHEELS}/rich_rst-2.1.0-py3-none-any.whl" \
    "${WHEELS}/cyclopts-4.22.5-py3-none-any.whl" \
    "${WHEELS}/ccimport-0.4.4-py3-none-any.whl" \
    "${WHEELS}/pccm-0.4.16-py3-none-any.whl" \
    "${WHEELS}/cumm_cu121-0.7.11-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/spconv_cu121-2.3.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/trimesh-4.5.1-py3-none-any.whl" \
    "${WHEELS}/texttable-1.7.0-py2.py3-none-any.whl" \
    "${WHEELS}/igraph-0.11.9-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/xatlas-0.0.11-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/scooby-0.11.2-py3-none-any.whl" \
    "${WHEELS}/vtk-9.6.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
    "${WHEELS}/pyvista-0.48.4-py3-none-any.whl" \
    "${WHEELS}/pymeshfix-0.17.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl"
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/glcontext-3.0.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/moderngl-5.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "${WHEELS}/plyfile-1.1.5-py3-none-any.whl" \
    "${WHEELS}/wadler_lindig-0.1.7-py3-none-any.whl" \
    "${WHEELS}/jaxtyping-0.2.38-py3-none-any.whl" \
    "${WHEELS}/typeguard-4.6.0-py3-none-any.whl"
}

if [[ ! -f "${STATE}/wheels.ok" ]]; then
  install_wheels
  touch "${STATE}/wheels.ok"
fi

# Packages added after the first wheel set was validated use their own marker,
# so an interrupted/retried offline install cannot incorrectly skip them.
if [[ ! -f "${STATE}/runtime-support-v2.ok" ]]; then
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/pooch-1.9.0-py3-none-any.whl"
  touch "${STATE}/runtime-support-v2.ok"
fi

if [[ ! -f "${STATE}/runtime-support-v3.ok" ]]; then
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/optree-0.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
  touch "${STATE}/runtime-support-v3.ok"
fi

if [[ ! -f "${STATE}/runtime-support-v4.ok" ]]; then
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/astor-0.8.1-py2.py3-none-any.whl"
  touch "${STATE}/runtime-support-v4.ok"
fi

if [[ ! -f "${STATE}/runtime-support-v5.ok" ]]; then
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/easydict-1.13-py3-none-any.whl"
  touch "${STATE}/runtime-support-v5.ok"
fi

if [[ ! -f "${STATE}/runtime-support-v6.ok" ]]; then
  "${PY}" -m pip install --no-index --no-deps \
    "${WHEELS}/lightning-2.3.3-py3-none-any.whl"
  touch "${STATE}/runtime-support-v6.ok"
fi

extract_source() {
  local archive=$1 destination=$2
  if [[ ! -f "${destination}/.extracted" ]]; then
    rm -rf -- "${destination}"
    mkdir -p "${destination}"
    tar -xzf "${archive}" --strip-components=1 -C "${destination}"
    touch "${destination}/.extracted"
  fi
}

extract_source "${WHEELS}/iopath-0.1.10.tar.gz" "${BUILD}/iopath"
extract_source "${SOURCES}/fvcore-0.1.5.post20221221.tar.gz" "${BUILD}/fvcore"
extract_source "${SOURCES}/pytorch3d-75ebeeaea0908c5527e7b1e305fbc7681382db47.tar.gz" "${BUILD}/pytorch3d"
extract_source "${SOURCES}/utils3d-3913c65d81e05e47b9f367250cf8c0f7462a0900.tar.gz" "${BUILD}/utils3d"
extract_source "${SOURCES}/moge-a8c37341bc0325ca99b9d57981cc3bb2bd3e255b.tar.gz" "${BUILD}/moge"
extract_source "${SOURCES}/gsplat-2323de5905d5e90e035f792fe65bad0fedd413e7.tar.gz" "${BUILD}/gsplat"

install_source() {
  local marker=$1 source=$2
  if [[ ! -f "${STATE}/${marker}.ok" ]]; then
    "${PY}" -m pip install --no-index --no-deps --no-build-isolation "${source}"
    touch "${STATE}/${marker}.ok"
  fi
}

install_source iopath "${BUILD}/iopath"
install_source fvcore "${BUILD}/fvcore"
install_source utils3d "${BUILD}/utils3d"
install_source moge "${BUILD}/moge"

if ! "${PY}" -c 'import pytorch3d' >/dev/null 2>&1; then
  install_source pytorch3d "${BUILD}/pytorch3d"
fi
if ! "${PY}" -c 'import gsplat' >/dev/null 2>&1; then
  # Stage-1 latent extraction only needs gsplat's Python API to be importable;
  # rasterization is never called.  Avoid compiling an unused CUDA extension
  # against the cluster's 12.6 compiler and the cloned cu121 PyTorch runtime.
  BUILD_NO_CUDA=1 install_source gsplat "${BUILD}/gsplat"
fi

"${PY}" - <<'PY'
import importlib
import json
import torch

required = [
    "cv2", "decord", "open3d", "xformers", "hydra", "omegaconf",
    "pytorch3d", "spconv", "trimesh", "optree", "astor", "easydict", "lightning",
    "utils3d", "moge", "gsplat",
    "sam3d_objects",
]
versions = {}
for name in required:
    module = importlib.import_module(name)
    # Some packages (notably utils3d) implement dynamic __getattr__ and treat
    # an innocent version lookup as a submodule import.  Read the module dict
    # directly so validation cannot create a false missing-module failure.
    versions[name] = vars(module).get("__version__", "unknown")
assert torch.cuda.is_available()
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
assert InferencePipelinePointMap.__name__ == "InferencePipelinePointMap"
x = torch.arange(8, device="cuda").float()
assert float(x.sum()) == 28.0
print("SAM3D_OFFLINE_ENV_OK", json.dumps({
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0), "modules": versions,
}, sort_keys=True))
PY

touch "${ENV}/.sam3d_runtime_ready"
rm -rf -- "${TMP}"
echo "end=$(date -Is)"
