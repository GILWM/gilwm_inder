# Release preparation: semantic and action models

This branch imports the deployed temporal-action implementation and its tests.
It retains the existing semantic preprocessing and inference tools. The source
snapshot was collected on 2026-09-16; no remote training directory was modified.

## Model identities

| Artifact | Conditioning | Role |
|---|---|---|
| Hybrid 1k | First-frame semantic cache | Curated presentation examples |
| Hybrid 4k | First-frame semantic cache, optionally disabled | September Track 1 generation and ablations |
| Semantic complete106k 5k | Complete first-frame semantic cache | Semantic model and action initialization |
| Action complete106k 5k | Semantic cache plus normalized 14D HDF5 trajectory | Temporal-context action continuation |

These are distinct checkpoints. A score or video from one is not evidence for
another. Fixed93, fixed121, native-length, and retimed videos are separate settings.

## Environment and data preparation

Use the MUSA environment described in the main README and its recorded package
snapshot. This is not a validated CUDA or macOS training port. Provide a MUSA
runtime image, Bash 4+, Docker, and the runtime workspace layout expected by
`projects/sam3d/scripts/run_cosmos_sam3d_musa.sh`:

```
envs/cosmos-musa/bin/python
weights/cosmos/models/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth
weights/sam3/sam3.pt
third_party/sam3-musa/
```

Also provision the original Cosmos text/vision assets, SAM 3, SAM 3D Objects,
and MoGe dependencies under their respective terms. An EMA checkpoint is not
a self-contained runtime. Preserve the repository's LICENSE and ATTRIBUTIONS.

Follow the existing README preprocessing workflow to create first-frame mask,
metadata, geometry, shape, and pose sidecars. Teacher video tokens are disabled
for these recipes. The training manifest must resolve its video, caption, and
action HDF5 paths, including symlink targets. The action loader reads
`joint_action/{left_arm,left_gripper,right_arm,right_gripper}` in that order.
Use `audit_worldarena_action_hdf5.py --help` and
`audit_video_action_alignment.py --help` to audit inputs before training.

Do not recompute normalization on evaluation trajectories. Use the statistics
associated with the trained model. The prepared action recipe uses the 105,498
eligible-item whitelist, a filtered subset of the semantic complete-condition
whitelist (106,354 items). These lists and the underlying data are not bundled.

## Training

Copy `projects/sam3d/configs/release.env.example` outside version control, fill in
the paths, runtime image and rendezvous host, and source it. Configure MCCL for
your network; historical host interface names are not portable defaults.

```
bash projects/sam3d/train_complete106k.sh semantic NODE_RANK
bash projects/sam3d/train_complete106k.sh action NODE_RANK
```

Run one launcher per node. Defaults reproduce four nodes, eight devices per node,
batch four per device, 93 uniformly sampled frames at 640×480, LR 1e-5, 500 warm-up
steps and 5,000 updates. For another topology, set `NNODES`, `NPROC_PER_NODE` and
`FSDP_SHARD_SIZE` consistently; changing topology is not an exact reproduction.

For semantic continuation, initialize from the historical Mix121k 5k checkpoint.
For action continuation, initialize from semantic complete106k 5k. The four
release EMA artifacts do not include the earlier Mix121k initialization or the
full hybrid training ancestry. Recreating those historical runs requires those
additional initializations; substituting a pretrained Cosmos checkpoint starts
a different experiment. Existing `train_v93.sh` and hybrid initialization tools
remain available for earlier recipes.

`SAM3D_BASE_CHECKPOINT` accepts an existing DCP checkpoint directory or a
consolidated `.pt` initialization. EMA initialization starts a new continuation;
it does not resume optimizer/scheduler state. Use a new run name for a new run.

The action recipe uses radius-four temporal mixing, four actions per latent,
8192 hidden units, tanh output scaled by 0.01, zero output initialization and
0.1 action dropout. Auxiliary action and representation-alignment losses are
disabled. The semantic recipe explicitly disables the action branch.

## Inference

Semantic inference uses the existing `predict2_video2world_inference_2b_sam3d`
experiment. Action complete106k 5k must use
`predict2_video2world_inference_2b_sam3d_action_temporal`, not the older LayerNorm
action experiment. Use `prepare_sam3d_action_benchmark50_ar.py --help` to construct
inputs with explicit first frame, condition cache, action HDF5 and normalization
paths. Its benchmark helper expects 50 episodes; `examples/inference.py` can
consume a prepared one-sample JSON for an initial smoke run.

```
export ACTION_INFERENCE_JSON=/path/to/one-sample.json
export ACTION_INFERENCE_OUTPUT=/path/to/output
export ACTION_INFERENCE_VIDEO=/path/to/output/sample-name.mp4
export ACTION_INFERENCE_DEVICES=0
export ACTION_INFERENCE_GPUS=1
bash projects/sam3d/scripts/run_temporal_action_conditioned_inference_smoke.sh /path/to/action-checkpoint
```

The checkpoint folder may contain only `model_ema_bf16.pt`. Include external
first-frame/cache/HDF5 paths in `COSMOS_EXTRA_BIND_PATHS` so the container can read
them. The helper verifies a 93-frame 640×480 output. The input JSON controls fps; the preparation helper defaults to 30 fps.
For AR output use the appropriate inference JSON and verify its requested length
separately. Two full 93-frame chunks with one overlap yield 185 unique frames;
121 frames is a truncated prefix. Do not silently retime these outputs for scoring.

## Verification

Run the action alignment, HDF5 loading, dataset, temporal conditioner, model wiring,
semantic conditioning and REPA tests under `tests/sam3d/` in the supported runtime.
Local CPU checks cover utilities and launcher arguments; they do not establish
multi-node training or MUSA inference reproducibility. See the PR for checks
actually performed and remaining runtime limitations.
