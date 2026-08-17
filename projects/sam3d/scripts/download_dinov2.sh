#!/usr/bin/env bash
set -euo pipefail

url="${DINOV2_URL:-https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth}"
destination="${DINOV2_DESTINATION:-/datassd/morka/cosmos-sam3d-work/weights/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}"
expected_size="${DINOV2_EXPECTED_SIZE:-346378731}"
proxy="${DINOV2_PROXY:-http://127.0.0.1:17890}"
partial="${destination}.incomplete"

mkdir -p "$(dirname "${destination}")"

file_size() {
  if [[ -f "$1" ]]; then
    stat -c '%s' "$1"
  else
    printf '0\n'
  fi
}

current_size="$(file_size "${destination}")"
if [[ "${current_size}" == "${expected_size}" ]]; then
  echo "DINOv2 checkpoint already complete: ${destination} (${current_size} bytes)"
  exit 0
fi

if (( current_size > 0 && current_size < expected_size )); then
  partial_size="$(file_size "${partial}")"
  if (( partial_size < current_size )); then
    mv -f "${destination}" "${partial}"
  else
    rm -f "${destination}"
  fi
elif (( current_size > expected_size )); then
  echo "Refusing unexpected oversized destination: ${destination} (${current_size} bytes)" >&2
  exit 1
fi

for attempt in $(seq 1 100); do
  current_size="$(file_size "${partial}")"
  echo "DINOv2 download attempt ${attempt}/100, resume from ${current_size}/${expected_size} bytes"

  if (( current_size == expected_size )); then
    break
  fi
  if (( current_size > expected_size )); then
    echo "Partial file is larger than expected: ${partial} (${current_size} bytes)" >&2
    exit 1
  fi

  HTTPS_PROXY="${proxy}" HTTP_PROXY="${proxy}" \
    curl --fail --location --continue-at - \
      --connect-timeout 30 --retry 5 --retry-delay 2 --retry-all-errors \
      --output "${partial}" "${url}" || true

  new_size="$(file_size "${partial}")"
  if (( new_size <= current_size )); then
    echo "No progress in attempt ${attempt}; retrying shortly" >&2
    sleep 2
  fi
done

current_size="$(file_size "${partial}")"
if [[ "${current_size}" != "${expected_size}" ]]; then
  echo "DINOv2 download incomplete: ${current_size}/${expected_size} bytes" >&2
  exit 1
fi

mv -f "${partial}" "${destination}"
echo "DINOv2 checkpoint complete: ${destination} (${current_size} bytes)"
