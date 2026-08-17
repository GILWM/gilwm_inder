#!/usr/bin/env bash
set -euo pipefail

pattern=${1:?usage: monitor_musa_job_memory.sh COMMAND_PATTERN OUTPUT_LOG}
output=${2:?usage: monitor_musa_job_memory.sh COMMAND_PATTERN OUTPUT_LOG}
interval=${MONITOR_INTERVAL_SECONDS:-2}

: >"$output"
while docker ps --no-trunc --format '{{.Command}}' | grep -Fq -- "$pattern"; do
  printf 'TIMESTAMP %s\n' "$(date '+%F %T')" >>"$output"
  mthreads-gmi | grep -E '^[[:space:]]*[0-7][[:space:]]' >>"$output" || true
  sleep "$interval"
done
printf 'MONITOR_DONE %s\n' "$(date '+%F %T')" >>"$output"
