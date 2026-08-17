#!/usr/bin/env bash
set -euo pipefail

workspace=/datassd/morka/cosmos-sam3d-work
log_root="${workspace}/logs/cache-v93"
cache_root=/datahdd/mccxadmin/cosmos-sam3d-cache/core15k-legacy4k-v93

running=0
done_count=0
failed=0
completed=0
skipped=0
for shard in $(seq 0 31); do
  pidfile="${log_root}/shard-${shard}.pid"
  log="${log_root}/shard-${shard}.log"
  if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    running=$((running + 1))
  fi
  line=$(grep '^DONE shard=' "$log" 2>/dev/null | tail -1 || true)
  if [[ -n "$line" ]]; then
    done_count=$((done_count + 1))
    c=$(sed -n 's/.*completed=\([0-9][0-9]*\).*/\1/p' <<<"$line")
    s=$(sed -n 's/.*skipped=\([0-9][0-9]*\).*/\1/p' <<<"$line")
    f=$(sed -n 's/.*failed=\([0-9][0-9]*\).*/\1/p' <<<"$line")
    completed=$((completed + ${c:-0}))
    skipped=$((skipped + ${s:-0}))
    failed=$((failed + ${f:-0}))
  fi
done
files=$(find "$cache_root" -type f -name condition.pt 2>/dev/null | wc -l)
printf 'CACHE_JOBS running=%d done_shards=%d completed=%d skipped=%d failed=%d files=%d\n' \
  "$running" "$done_count" "$completed" "$skipped" "$failed" "$files"
