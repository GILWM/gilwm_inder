# Cosmos 原生 Action Expert 设计

## 结论

本实现保留 FlowWAM 的核心结构：动作 token 自注意力、逐层读取视频 DiT
特征、逐时刻预测动作。它不复刻 Wan 的 3072 维接口，而是原生使用
Cosmos 2B 的 2048 维特征，并在交叉注意力前压缩空间 token。

旧的 `TemporalActionAlignmentHead` 仍是默认值。只有显式设置
`ACTION_ARCHITECTURE=cosmos_action_expert` 才会构建新网络，因此旧配置、旧
checkpoint 和纯推理都不受影响。

## 张量流

以当前 93 帧、480x640 训练为例：

1. VAE 时间压缩后约为 24 个 latent 帧。
2. Cosmos 空间 patch 后，每个 DiT 中间层约为
   `[B, 24*30*40, 2048] = [B, 28800, 2048]`。
3. 从 DiT 的第 `[3, 7, 11, 15, 19, 23]` 层取特征；每帧恢复为真实
   `30x40` 网格，再自适应池化到 `2x2`。
4. 每个来源层的上下文变为 `[B, 24*4, 512] = [B, 96, 512]`，共享
   `2048 -> 512` 投影。
5. 固定时间 query 为 `[B, 24, 512]`，叠加位置编码和视频扩散 timestep
   编码。
6. 6 个 Action Expert block 依次执行动作自注意力、视频交叉注意力和
   FFN；第 i 个 block 对应读取一层 Cosmos 特征。
7. 输出 `[B, 24, 14]`，监督动作按相同时间轴从原始动作序列均匀取样。

动作真值不会作为 query 输入，只在 loss 端使用。这一点避免了把干净动作
送进网络后直接复制答案的问题；该网络仍然是 inverse-dynamics 监督，不是
推理时依赖未来动作的 action-conditioned 生成器。

默认网络为 512 hidden、8 heads、6 blocks、FFN 倍率 4，约 2700 万参数。
相比 FlowWAM 的约 7.78 亿 Action Expert，它保留了逐层视频读取能力，但将
参数量和交叉注意力 token 数控制在适合现有 80G 卡训练的范围内。

## Loss

```text
L_total = L_diffusion
        + action_loss_weight * (
              L_action_mse
            + action_alignment_weight * L_action_cosine
          )
```

- `L_action_mse`：预测的归一化 14D 动作与对应时刻真值的 MSE。
- `L_action_cosine`：最终 action token 与独立 action target encoder 的逐时刻
  cosine distance。
- 视频扩散 timestep 直接进入 action query，使同一个 head 能解释不同噪声
  强度下的 Cosmos 中间特征。

## 本次训练数据

训练入口合并以下两个独立 manifest，不复制视频或 HDF5：

- `core15k_filter_75_455`：13,487 条，其中 12,850 条不少于 93 帧。
- `robotwin2_filter_75_455`：25,930 条，全部不少于 93 帧。
- 93 帧训练实际可用合计：38,780 条。

两批数据的 manifest 均已为每条样本提供 14D `actions.hdf5`。多 manifest
加载器会以各 manifest 所在目录解析视频、instruction 和 action 相对路径。

### HDF5 归一化

当前网络只读取并监督 H5 中的 14 维 `joint_action`：左臂 6 维、左夹爪
1 维、右臂 6 维、右夹爪 1 维。每一维使用两批可训练数据的全局统计做
z-score：

```text
action_normalized[d] = (action[d] - mean[d]) / (std[d] + 1e-6)
```

统计覆盖 38,780 个有效 H5、7,828,045 个动作时刻。该处理不是 min-max，
不会把结果限制到 `[0,1]` 或 `[-1,1]`，也不做 clipping。H5 中的
`endpose/*`、相机参数、内嵌 RGB 和 pointcloud 当前不送入 Action Expert，
因此没有在这条训练链路中额外 normalize。

先生成与实际 >=93 帧训练集合严格一致的归一化统计：

```bash
python projects/sam3d/scripts/audit_worldarena_action_hdf5.py \
  /datahdd/mccxadmin/train_data/core15k_filter_75_455 \
  /datahdd/mccxadmin/train_data/robotwin2_filter_75_455 \
  --min-video-frames 93 \
  --skip-schema-audit --progress-every 1000 \
  --expected-count 38780 \
  --write-norm /datahdd/mccxadmin/train_data/filter_75_455_v93_action_norm_stats.npz \
  --json-report /datahdd/mccxadmin/train_data/filter_75_455_v93_action_norm_stats.json
```

训练使用：

```bash
# 仍需像原训练一样提供 COSMOS_SAM3D_WORKSPACE、SAM3D_BASE_CHECKPOINT、
# MASTER_ADDR 和各节点 NODE_RANK。
projects/sam3d/train_action_expert_filter_75_455.sh NODE_RANK
```

smoke test 可显式设置：

```bash
NNODES=1 MAX_ITER=3 SAVE_ITER=3 PER_DEVICE_BATCH_SIZE=1 \
  projects/sam3d/train_action_expert_filter_75_455.sh 0
```

两个筛选目录中的样本是指向原始数据的符号链接。启动脚本默认通过
`COSMOS_EXTRA_BIND_PATHS=/datahdd` 将链接目标一并挂入容器；如果数据迁移到
其他文件系统，可用冒号分隔的绝对路径覆盖该变量。

当前正式实验只训练 Core15K，固定 1000 iter。其入口为：

```bash
projects/sam3d/train_action_expert_core15k_1k.sh NODE_RANK
```

默认使用 4 节点、每节点 8 卡、每卡 batch size 4，并在 500/1000 iter 保存。
动作归一化统计固定来自官方 WorldArena 1000 题，加上本次真正可训练的
12,850 条 Core15K（`>=93` 帧）样本。Robotwin2 不参加这次训练，也不参与
这份 normalization 统计。

对应统计文件为：

```text
/datahdd/mccxadmin/train_data/official1000_core15k_v93_action_norm_stats.npz
```

审计结果为 13,850/13,850 个 H5 有效、0 失败，共 2,713,621 个动作时刻。

完整长任务的 `MAX_ITER`、batch size 和起始 checkpoint 应在 smoke test
通过后再确定。
