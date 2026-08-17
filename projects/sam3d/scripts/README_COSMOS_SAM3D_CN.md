# Cosmos Predict2.5 × SAM 3 / SAM 3D Objects

本文档描述独立实验分支的目录、数据流、模型改动、训练目标和复现入口。最终输出仍是 Cosmos VAE 解码得到的 MP4 视频；SAM 系列模型仅负责从输入首帧/教师帧提取额外条件。

## 1. 独立工作区

摩尔卡（MUSA，训练与视频推理）：

```text
/datassd/morka/cosmos-sam3d-work
├── cosmos-predict2.5-sam3d-musa/   # 独立 Cosmos 代码
├── envs/cosmos-musa/               # 独立 Python 环境
├── weights/
│   ├── cosmos/                     # Cosmos 2.5、VAE、Reason1
│   ├── sam3/sam3.pt
│   ├── sam3d-objects/checkpoints/
│   └── torch/hub/checkpoints/      # DINOv2
├── third_party/                    # SAM3、SAM3D Objects、DINOv2 源码
├── scripts/
├── configs/
├── training-outputs/
└── inference-outputs/
```

DSS（NVIDIA CUDA，只做原生 SAM 3D Objects 预处理）：

```text
/gpfs/share/home/2301213237/cosmos-sam3d-work
├── envs/sam3d-objects-offline/
├── weights/sam3d-objects/checkpoints/
├── weights/moge-vitl/model.pt
├── offline/                        # 离线依赖包和基环境归档
├── src/sam-3d-objects-official/
├── src/dinov2-official/            # 固定提交的官方 DINOv2 torch.hub 源码
├── scripts/
└── smoke-output/native-sidecars/
```

DSS 与摩尔卡之间不直传。视频、mask cache 和生成的 sidecar 都经本机中转。

DSS 的离线环境固定为 Python 3.10、PyTorch 2.5.1+cu121；自定义 CUDA 扩展使用集群的 GCC 12.3 和 CUDA 12.6 编译。不要使用节点默认 GCC 4.8，也不要安装要求 Python 3.11 的 `jaxtyping 0.3.x`；当前锁定为 `jaxtyping 0.2.38`。原生 sidecar 只运行 SAM 3D Objects stage-1，因此 stage-2 的 Kaolin/FlexiCubes/mesh renderer 采用懒加载，不进入运行时依赖链；这也避开了 DSS 系统 glibc 低于 Kaolin 预编译包要求的问题。完整离线依赖包为：

```text
/gpfs/share/home/2301213237/cosmos-sam3d-work/offline/sam3d-offline-bundle-v3.tgz
SHA256 ef4b99c54bd0bc2035ebb54dd229146b4bbb83126bffc1ed07e1b14082f566c4
```

该版本补齐了 `easydict==1.13`、`lightning==2.3.3` 和固定提交
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` 的官方 DINOv2 源码。SAM 3D
checkpoint 本身包含两套冻结 DINOv2 condition embedder 权重；离线启动时只从本地
源码构造同结构的 `dinov2_vitl14_reg(pretrained=False)`，随后由 checkpoint 严格加载
真实参数，因此不会额外下载 DINOv2 权重。

## 2. 条件数据

每条样本最终使用一个固定形状的 `condition.pt`：

| 字段 | 形状 | 来源/含义 |
|---|---:|---|
| `sam3d_tokens` | `[8,256,768]` | SAM 3D Objects 使用的冻结 DINOv2 教师特征 |
| `sam_masks` | `[8,120,160]` | SAM 3 文本提示实例分割 mask |
| `sam_mask_meta` | `[8,8]` | 框、置信度、面积、提示编号和有效位 |
| `sam3d_geometry` | `[8,120,160]` | 深度、XYZ、法线、有效性 |
| `sam3d_shape_latents` | `[8,256,8]` | 原生 SAM 3D Objects stage-1 shape latent |
| `sam3d_object_pose` | `[8,10]` | translation(3)、quaternion(4)、scale(3) |

处理流程：

1. MUSA 从视频均匀抽取 8 个教师帧，强制保留首帧。
2. DINOv2 对教师帧生成 token；SAM3 对首帧产生显式 mask 和元信息。
3. DSS 使用同一个首帧和同一组 mask，运行原生 SAM 3D Objects stage-1，产生 shape latent、pose 和 point map。
4. sidecar 经本机回传 MUSA；`merge_sam3d_object_sidecars.py` 只合并原生 3D 字段，不重跑 SAM3/DINO，避免条件错位。

传给官方 pipeline 的 mask 必须是 `uint8` 的 `0/255` alpha。布尔 mask 会被官方
`image_to_float` 再除以 255，前景变成 `1/255`，最终被 point-map normalizer 当作空
mask；`prepare_sam3d_object_sidecars.py` 已在接口边界完成显式转换。

缓存路径：

```text
CACHE_ROOT/<batch>/<question_id>/condition.pt
```

## 3. DiT 改动

Cosmos Reason1 文本条件仍走原来的投影路径。六类 SAM/SAM3D 条件分别经过独立 tokenizer/projector，附加模态 embedding，再拼成最多 256 个结构化条件 token。

每个 DiT block 增加一条复用原生 cross-attention 投影的条件注意力残差：

```text
DiT hidden ── cross-attend(SAM multi-token context) ── gate ── residual add
```

每层 gate 从 0 初始化，因此加载原始 Cosmos 权重时，step 0 的输出与未改动模型一致；训练开始后 gate、条件 projector 和 LoRA 一起学习。

## 4. PAIWorld 适配

PAIWorld 面向多视角，本实验当前是单视角视频，因此没有照搬它依赖相机内外参的跨视角 Geo-RoPE。保留并适配了与本任务直接对应的两部分：

- 零初始化的显式信息通路：原论文用跨视角 attention，本分支用 SAM/SAM3D 条件 cross-attention。
- Latent 3D-REPA：抽取第 13 层 DiT 中间 token，与冻结 DINO 教师 token 对齐；分别计算帧内空间关系和跨帧时间关系。

对齐比较的是 anchor-sampled cosine relation，而不是强制两个编码器逐 token 数值相等：

```text
L_REPA = SmoothL1(S_spatial_DiT, S_spatial_teacher)
       + SmoothL1(S_temporal_DiT, S_temporal_teacher)

L_total = L_diffusion + 0.5 * L_REPA
```

这与 PAIWorld 的关系蒸馏形式和 `lambda=0.5` 一致，同时允许 SAM 3D Objects latent、显式 mask 和几何作为生成条件参与 DiT。

## 5. 预处理与校验

MUSA 生成 SAM3 mask + DINO cache：

```bash
/datassd/morka/cosmos-sam3d-work/scripts/run_condition_preprocess_musa.sh \
  /datassd/morka/cosmos-sam3d-work/scripts/prepare_sam3d_conditions.py \
  --limit 1 --overwrite \
  --cache-root /datahdd/mccxadmin/cosmos-sam3d-cache/real-smoke
```

DSS 原生 stage-1 smoke 由 Slurm 入口运行：

```bash
sbatch /gpfs/share/home/2301213237/cosmos-sam3d-work/scripts/run_sam3d_sidecar_smoke_dss.sbatch
```

sidecar 回传 MUSA 后合并：

```bash
/datassd/morka/cosmos-sam3d-work/scripts/run_cosmos_sam3d_musa.sh \
  /datassd/morka/cosmos-sam3d-work/scripts/merge_sam3d_object_sidecars.py \
  --base-cache-root /datahdd/mccxadmin/cosmos-sam3d-cache/real-smoke \
  --sidecar-root /datahdd/mccxadmin/cosmos-sam3d-sidecars/native-smoke \
  --output-root /datahdd/mccxadmin/cosmos-sam3d-cache/native-smoke \
  --overwrite
```

完整条件必须用严格模式校验：

```bash
/datassd/morka/cosmos-sam3d-work/scripts/run_cosmos_sam3d_musa.sh \
  /datassd/morka/cosmos-sam3d-work/scripts/validate_sam3d_cache.py \
  /datahdd/mccxadmin/cosmos-sam3d-cache/native-smoke \
  --require-nonzero-tokens \
  --require-masks \
  --require-geometry \
  --require-sam3d-objects
```

## 6. 训练

完整实验默认配置：121 帧、`480×640`、batch size 1、30,000 iter、AdamW、峰值学习率 `3e-5`、3,000 iter warmup、cosine decay、REPA 权重 `0.5`。

8 卡 smoke：

```bash
SAM3D_CACHE_ROOT=/datahdd/mccxadmin/cosmos-sam3d-cache/native-smoke \
SAM3D_RUN_NAME=smoke-full-native \
SAM3D_MAX_ITER=2 \
/datassd/morka/cosmos-sam3d-work/scripts/run_sam3d_real_smoke.sh
```

正式训练入口：

```bash
/datassd/morka/cosmos-sam3d-work/scripts/run_cosmos_sam3d_musa.sh \
  -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_video2world_training_2b_sam3d_full \
  job.wandb_mode=disabled
```

## 7. 视频推理

官方推理参数新增：

```json
"sam3d_condition_path": "/path/to/condition.pt"
```

推理入口会检查六类张量的维度和有限值，给它们增加 batch 维后送入自定义 conditioner。CFG 条件分支使用完整 SAM3D 条件，负条件分支按各模态 dropout 规则清零。

DCP checkpoint 先转成 EMA BF16，再用 2 卡 context parallel 单 chunk 生成 121 帧视频。
这里的 2 卡不是显存妥协：121 帧对应 31 个时间 latent，`480×640` 的 latent 高度只
能再承接一个因子 2；CP=4/8 会继续切宽度，而 Cosmos 官方明确禁止宽度切分以保护
画质。因此该分辨率/帧数下最大合法 CP 是 2。

```bash
/datassd/morka/cosmos-sam3d-work/scripts/run_sam3d_inference_smoke.sh \
  /path/to/checkpoints/iter_000000002
```

脚本最终必须打印：

```text
SAM3D_VIDEO_INFERENCE_OK /.../legacy4k_00000001_sam3d.mp4
```

## 8. 已通过的验证

- 独立 Cosmos 代码、环境和权重可以脱离旧目录加载。
- SAM3 checkpoint 的 1,465 个 tensor key 和 861,235,128 个参数可在 MUSA 安全加载。
- SAM3 在 MUSA 上完成图像编码和文本提示推理。
- DINO-only 与真实 SAM mask cache 均通过固定形状、有限值和非零检查。
- 多 token DiT 单测证明：gate=0 时严格保持原输出，打开 gate 后条件会改变输出。
- 8 卡 FSDP 已完成真实 SAM mask 条件的一次前向、反向、优化器更新、DCP 保存，并干净打印 `Done with training`。
- 推理 cache loader 已用真实 `condition.pt` 验证六类张量接口。
- DSS 原生 stage-1 已用官方权重完成 25 步 sparse-structure 采样；sidecar 包含非零的
  `[1,4096,8]` shape latent、`[8,120,160]` 几何和 `[1,10]` 位姿。
- 原生 sidecar 经 DSS→本机→MUSA 中转后，完整 cache 已通过 token、mask、geometry
  和 native object 四项严格非零校验。
- 8 卡完整条件训练 `smoke-full-native-r1` 已完成 2 次前向、反向和优化器更新，loss
  为 `0.1907 → 0.1911`，并写出 `iter_000000002` DCP。
- 转换后的 EMA BF16 checkpoint 含 717 个 tensor，其中 28 个属于 SAM/SAM3D 新通路；
  28 个逐层 gate 已从严格的全零初始化更新为非零值。
- 训练后权重已在 CP=2 下完成 4-step、单 chunk、121 帧推理；Decord 实测输出为
  `640×480`、恰好 121 帧：

```text
/datassd/morka/cosmos-sam3d-work/inference-outputs/native-smoke-cp2/legacy4k_00000001_sam3d.mp4
```

以上验证覆盖了原生 sidecar、严格完整 cache、全条件训练、训练后 checkpoint 参数和
最终 MP4。正式扩大数据规模时复用同一条预处理/训练入口即可，不需要再改变模型接口。

## 9. 最终归档与校验

摩尔卡上的完整 SAM 3D Objects 权重目录为：

```text
/datassd/morka/cosmos-sam3d-work/weights/sam3d-objects/checkpoints
```

目录内的 `SAM3D_OBJECTS_SHA256.txt` 固化了全部正式 checkpoint、配置文件及空占位
文件的 SHA256。可在该目录直接执行：

```bash
sha256sum -c SAM3D_OBJECTS_SHA256.txt
```

2026-08-12 的最终归档结果为 16 项全部 `OK`。最后补传的
`slat_generator.ckpt` 大小为 `4,906,537,684` 字节，SHA256 为
`91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb`。

可离线重建 DSS 原生 SAM 3D stage-1 环境的最终依赖包也已同时归档到摩尔卡：

```text
/datassd/morka/cosmos-sam3d-work/offline/sam3d-offline-bundle-v3.tgz
SHA256 ef4b99c54bd0bc2035ebb54dd229146b4bbb83126bffc1ed07e1b14082f566c4
```

可直接用于后续训练/推理的本次验证 checkpoint 与视频分别是：

```text
/datassd/morka/cosmos-sam3d-work/training-outputs/smoke-full-native-r1/cosmos_predict_v2p5_sam3d/video2world/smoke-full-native-r1/checkpoints/iter_000000002
/datassd/morka/cosmos-sam3d-work/training-outputs/smoke-full-native-r1/cosmos_predict_v2p5_sam3d/video2world/smoke-full-native-r1/checkpoints/iter_000000002/model_ema_bf16.pt
/datassd/morka/cosmos-sam3d-work/inference-outputs/native-smoke-cp2/legacy4k_00000001_sam3d.mp4
```
