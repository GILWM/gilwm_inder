#!/usr/bin/env bash
set -euo pipefail

workspace=/datassd/morka/cosmos-sam3d-work
image=10.200.53.208/ci/nanhu-computing-framework:v2.1.6-rc1
name=cosmos-sam3-download
log="${workspace}/logs/sam3-download.log"
pid_file="${workspace}/logs/sam3-download.pid"

if docker inspect "${name}" >/dev/null 2>&1; then
  echo "container ${name} already exists" >&2
  exit 1
fi

: >"${log}"
nohup docker run --rm \
  --name "${name}" \
  --network host \
  -e HTTP_PROXY=http://127.0.0.1:17890 \
  -e HTTPS_PROXY=http://127.0.0.1:17890 \
  -e NO_PROXY=127.0.0.1,localhost \
  -v /datassd:/datassd \
  "${image}" \
  bash "${workspace}/scripts/download_meta_weights.sh" \
    facebook/sam3 "${workspace}/weights/sam3" 2 sam3.pt \
  >>"${log}" 2>&1 &
echo "$!" >"${pid_file}"
echo "STARTED pid=$(cat "${pid_file}") log=${log}"
