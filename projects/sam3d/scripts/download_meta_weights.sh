#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 REPOSITORY DESTINATION [CREDENTIAL_INDEX] [FILE ...]" >&2
  exit 2
fi

repository="$1"
destination="$2"
credential_index="${3:-1}"
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi
files=("$@")
workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
token_file="${HF_TOKEN_FILE:-${workspace}/.secrets/hf_token}"

if [[ ! -s "${token_file}" ]]; then
  echo "Hugging Face token file is missing or empty" >&2
  exit 1
fi

# The shared token file may contain several independently managed credentials.
# Use one complete line and never place the token in argv or logs.
HF_TOKEN="$(sed -n "${credential_index}p" "${token_file}")"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "Credential index ${credential_index} does not exist" >&2
  exit 1
fi
export HF_TOKEN
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# Direct HTTP is slower on an unconstrained link, but is much more reliable
# behind the cluster's localhost HTTP proxy than the Xet multi-stream client.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
unset HF_XET_HIGH_PERFORMANCE
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "${destination}"
download_args=(
  --repo-type model
  --local-dir "${destination}"
  --max-workers 1
  "${repository}"
)
if [[ ${#files[@]} -gt 0 ]]; then
  download_args+=("${files[@]}")
fi
max_attempts="${HF_DOWNLOAD_ATTEMPTS:-200}"
attempt=1
while true; do
  if hf download "${download_args[@]}"; then
    break
  fi
  if (( attempt >= max_attempts )); then
    echo "Download failed after ${attempt} resumable attempts" >&2
    exit 1
  fi
  echo "Download connection interrupted; resuming (attempt ${attempt}/${max_attempts})" >&2
  attempt=$((attempt + 1))
  sleep 5
done

du -sh "${destination}"
find "${destination}" -type f -printf '%s %P\n' | sort -k2 > "${destination}/files.size-manifest.txt"
echo "DOWNLOAD_COMPLETE ${repository} ${destination}"
