#!/usr/bin/env bash
set -euo pipefail

workspace=/datassd/morka/cosmos-sam3d-work
log="${workspace}/logs/runtime-copy.log"
pid_file="${workspace}/logs/runtime-copy.pid"

if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "runtime copy is already running: pid=$(cat "${pid_file}")"
  exit 0
fi

: >"${log}"
nohup bash "${workspace}/scripts/clone_runtime.sh" >>"${log}" 2>&1 &
echo "$!" >"${pid_file}"
echo "STARTED pid=$(cat "${pid_file}") log=${log}"
