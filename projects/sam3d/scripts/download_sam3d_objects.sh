#!/usr/bin/env bash
set -euo pipefail

workspace="${COSMOS_SAM3D_WORKSPACE:-/datassd/morka/cosmos-sam3d-work}"
repository="${SAM3D_OBJECTS_REPOSITORY:-tuandao-zenai/sam-3d-objects}"
destination="${workspace}/weights/sam3d-objects"
marker="${destination}/.download-complete"

mkdir -p "${destination}"
if [[ -f "${marker}" ]]; then
  echo "SAM 3D Objects weights already marked complete: ${destination}"
  exit 0
fi

for attempt in $(seq 1 100); do
  echo "SAM 3D Objects download attempt ${attempt}/100 from ${repository}"
  if hf download \
      --repo-type model \
      --local-dir "${destination}" \
      --max-workers 4 \
      "${repository}"; then
    required=(
      checkpoints/pipeline.yaml
      checkpoints/ss_generator.ckpt
      checkpoints/ss_decoder.ckpt
      checkpoints/slat_generator.ckpt
      checkpoints/slat_decoder_gs.ckpt
      checkpoints/slat_decoder_mesh.ckpt
    )
    missing=0
    for relative in "${required[@]}"; do
      if [[ ! -s "${destination}/${relative}" ]]; then
        echo "Missing required artifact after download: ${relative}" >&2
        missing=1
      fi
    done
    if [[ "${missing}" == "0" ]]; then
      printf 'repository=%s\ncompleted_at=%s\n' "${repository}" "$(date --iso-8601=seconds)" > "${marker}"
      echo "SAM 3D Objects weights complete: ${destination}"
      exit 0
    fi
  fi
  echo "Download interrupted; cached chunks will be resumed" >&2
  sleep 3
done

echo "SAM 3D Objects download did not complete after 100 attempts" >&2
exit 1
