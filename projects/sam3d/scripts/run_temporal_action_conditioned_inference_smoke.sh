#!/usr/bin/env bash
set -euo pipefail

# Structural pairing is mandatory: temporal-action checkpoints contain a
# mixer and were trained with tanh output, so they must not use the legacy
# LayerNorm action experiment.
export SAM3D_ACTION_INFERENCE_EXPERIMENT=predict2_video2world_inference_2b_sam3d_action_temporal
exec "$(dirname "$0")/run_action_conditioned_inference_smoke.sh" "$@"
