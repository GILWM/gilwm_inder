# Cosmos + SAM3D 的 Action-Conditioned 生成

## 结论

当前版本把 action 作为视频生成条件，同时用于训练和推理。它不再把 action 仅作为 inverse-dynamics 的预测目标：归一化后的 14 维 action 会进入 Cosmos DiT 的 timestep embedding，因此相同首帧、提示词和随机种子在使用不同 action 时可以生成不同运动。

旧的 Action Expert 预测头仍保留用于 checkpoint 兼容和可选消融，但正式 action-conditioned 训练默认设置 `ACTION_LOSS_WEIGHT=0`，总目标就是原本的 rectified-flow 视频扩散 loss。这样没有额外的 action MSE/cosine loss，也不会让一个辅助目标压过视频生成目标。

## 时间对齐与张量形状

当前训练使用 93 个像素帧。因果 VAE 进行 4 倍时间压缩后得到 24 个 latent 帧：

```text
pixel/action: 0 | 1 2 3 4 | 5 6 7 8 | ... | 89 90 91 92
latent:       0 |    1    |    2    | ... |      23
```

- action 与视频先使用完全相同的源帧整数索引抽取，得到 `[B,93,14]`。
- 第 0 帧已经是观测到的首帧，不需要 action 生成；第 1～4 帧的 action 拼为 latent 1 的条件，第 5～8 帧对应 latent 2，以此类推。
- 每组 4 个 action 先拼成 56 维，经 MLP 得到 `[B,23,2048]` 的未来 timestep 增量；在首帧位置补零后成为 `[B,24,2048]`。当前稳定版本不生成或注入额外的 AdaLN 增量。
- action MLP 正常初始化，输出先做无参数 LayerNorm，再乘固定的 `action_conditioning_scale=0.01`，加入每个 latent 时刻的 timestep embedding，随后继续经过 Cosmos 自己的 timestep RMSNorm。这里不用“从 0 学起”的门控，也不把 6144 维 action 残差绕过归一化直接加到每层 AdaLN：这两种路径在 MUSA FSDP2 大 batch 下会触发 peer-rank NaN。约 1900 万参数的 action MLP 作为独立 FSDP 单元分片。
- 归一化 action 在进入 MLP 前截到 `[-10,10]`。全量 Core15K 的 34,757,982 个值中只有 347 个超过该范围（约 0.001%），用于阻断少量离群值，不改变绝大多数数据。
- 启用 action conditioning 后，如果训练或推理没有提供 action，或者长度不是 `1 + (latent_T-1)*4`，程序会直接报错，不能静默退化为无 action 模式。

## HDF5 读取与归一化

只读取官方四项并按下列顺序组成 14 维向量：

```text
joint_action/left_arm       6D
joint_action/left_gripper   1D
joint_action/right_arm      6D
joint_action/right_gripper  1D
```

`joint_action/vector`、`endpose/*`、RGB、相机参数和 pointcloud 等冗余字段全部忽略。每一维使用固定统计做 z-score：

```text
normalized[d] = (action[d] - mean[d]) / (std[d] + 1e-6)
```

当前 Core15K 训练和推理共用：

```text
/datahdd/mccxadmin/train_data/official1000_core15k_v93_action_norm_stats.npz
```

该统计来自官方 WorldArena 1000 题加上可训练的 Core15K 子集。不要在推理时重新计算统计。

## 对齐审计结果

- Core15K manifest 共 13,487 条，其中 12,850 条不少于 93 帧并进入训练。
- 12,850/12,850 的 manifest 视频帧数与官方 action 长度相等，并全部通过“视频抽帧索引等于 action 索引”的全量检查。
- 随机实际解码 128 个 MP4，128/128 的真实帧数、manifest 帧数和 action 长度一致。
- 全量报告：`/datassd/morka/cosmos-sam3d-work/action-alignment-audit-core15k-all-20260820.json`。
- 实际解码报告：`/datassd/morka/cosmos-sam3d-work/action-alignment-audit-core15k-128-20260820.json`。

## 训练

Core15K 入口：

```bash
projects/sam3d/train_action_expert_core15k_1k.sh NODE_RANK
```

重要环境变量：

```bash
ACTION_CONDITIONING_ENABLED=true
ACTION_CONDITIONING_HIDDEN_DIM=8192
ACTION_LOSS_WEIGHT=0
NUM_FRAMES=93
PER_DEVICE_BATCH_SIZE=4
MAX_ITER=5000
SAVE_ITER=500
```

启动器默认使用 4 台机器、每台 8 卡，并依据 `ens11np0` 自动解析每台机器自己的 RDMA HCA（wx25 为 `mlx5_0`，wx26～wx28 为 `mlx5_2`）。

## 推理

每个 JSON/JSONL 推理样本增加两项：

```json
{
  "action_hdf5_path": "/path/to/actions.hdf5",
  "action_norm_path": "/datahdd/mccxadmin/train_data/official1000_core15k_v93_action_norm_stats.npz"
}
```

推理会读取官方 14D action，按 `num_output_frames` 均匀映射到生成时间轴并保留第一项。单 chunk 会一次传入完整 action；AR 会按照每个 chunk 的绝对帧区间切片，重叠帧也使用相同绝对时刻的 action。CFG 只改变文本/视频条件的 dropout，action 在正、负分支中保持一致。

加载 action checkpoint 时必须选择实验 `predict2_video2world_inference_2b_sam3d_action`。这个实验会实例化与训练完全相同的 14D→8192→2048 action timestep MLP；仅在 JSON 里填写 `action_hdf5_path`、却使用普通 SAM3D 实验，会导致 action checkpoint 参数没有对应模块可加载。

## 回归测试

`tests/sam3d` 覆盖：

- 视频与 action 精确使用同一组抽帧索引；
- 冗余 H5 字段不影响官方字段读取；
- action chunk 严格为 `[1:5]、[5:9]…`，拒绝 92/94 等 off-by-one 长度；
- `action_conditioning_scale=0` 时 action 分支严格保持原模型输出；
- 固定小批次可把条件分支 loss 降到初始值 5% 以下；
- eval/推理路径确实传入 action，改变 action 会改变预测。
