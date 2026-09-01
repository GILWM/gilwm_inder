#!/usr/bin/env bash
set -euo pipefail

root=/bdsys_hdd/datassd/morka/cosmos-work/eval/sam3d_complete106k_every500_balanced50_ar93_overlap01_640x480_121f_20260901
state=${root}/state
rm -f "${state}/ALL_DONE" "${state}/ALL_FAILED" "${state}/FINAL_SUMMARY.txt"

while true; do
  failed=$(find "${state}" -maxdepth 2 -name '*.FAILED' -type f | wc -l)
  done_count=$(find "${state}" -maxdepth 1 -name 'queue_[0-3].DONE' -type f | wc -l)
  videos=$(find "${root}" -path '*/videos/*.mp4' -type f -size +0c | wc -l)
  valid=$(find "${root}" -path '*/state/VALID' -type f | wc -l)
  proofs=$(find "${state}" -mindepth 2 -maxdepth 2 -name CONDITION_PROOF_VALID -type f | wc -l)
  printf 'FINALIZE_PROGRESS queues=%s/4 videos=%s/1000 valid=%s/20 proofs=%s/10 failed=%s time=%s\n' \
    "${done_count}" "${videos}" "${valid}" "${proofs}" "${failed}" "$(date --iso-8601=seconds)"
  if (( failed > 0 )); then
    touch "${state}/ALL_FAILED"
    exit 1
  fi
  if (( done_count == 4 )); then
    if (( videos != 1000 || valid != 20 || proofs != 10 )); then
      touch "${state}/ALL_FAILED"
      exit 1
    fi
    printf 'status=complete\nvideos=1000\nexperiments=20\ncondition_proofs=10\ncompleted_at=%s\n' \
      "$(date --iso-8601=seconds)" >"${state}/FINAL_SUMMARY.txt"
    touch "${state}/ALL_DONE"
    exit 0
  fi
  sleep 30
done
