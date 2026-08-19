# WorldArena H5、FlowWAM 与动作监督接入说明

## 结论

`dataset_track1` 的动作字段可以直接复用，但整份数据不能原样作为 FlowWAM 的视频训练集。

- 动作兼容：两者都使用 14 维绝对关节状态，顺序为
  `left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)`。
- 视频不兼容：FlowWAM 原始 dataloader 要求每个 HDF5 内存在
  `observation/head_camera/rgb`、`observation/left_camera/rgb`、
  `observation/right_camera/rgb`。公开的 WorldArena Track 1 H5 没有这些字段，只有动作、末端位姿和空 pointcloud。
- 目录不兼容：FlowWAM 原生目录是
  `<root>/<task>/<variant>/data/episode*.hdf5`；WorldArena 是
  `<root>/data/<task>/episode*.hdf5`。
- instruction 不兼容：FlowWAM 读取 `{"seen": [...]}`，WorldArena 是
  `{"instruction": "..."}`。
- 公开 Track 1 只有首帧 PNG，没有逐帧参考视频。因此它可以用于动作格式验证、首帧条件推理，不能单独完成视频 latent 与动作的联合训练。

本仓库采用“视频主数据 + WorldArena H5 动作 sidecar”的方式接入，不要求把 RGB 重复写入 H5。

## 训练数据配对格式

视频数据根目录仍沿用本仓库的紧凑格式，并在 `manifest.jsonl` 中增加
`trajectory_hdf5`。该路径相对于 `ACTION_HDF5_ROOT`：

```json
{"batch":"robotwin","video":"robotwin/robotwin_00000001/video.mp4","instruction":"robotwin/robotwin_00000001/instruction.json","trajectory_hdf5":"data/fixed_scene_task/episode1.hdf5"}
```

也支持无需 manifest 显式配对的同名形式，例如视频名为 `episode1.mp4`，
H5 为 `data/fixed_scene_task/episode1.hdf5`。正式训练更推荐 manifest，避免不同任务下同名 episode 产生歧义。

必须保证视频和 H5 描述同一条轨迹。代码会把视频实际抽中的源帧时间戳映射到动作时间轴；即使二者总长度不同，也按相对时间对齐。若控制记录相对图像存在固定滞后，可设置：

```bash
export ACTION_ALIGNMENT_OFFSET=1  # 正数表示监督时取更晚的动作
```

## 模型与 loss

动作监督默认关闭，因而旧训练配置、旧 checkpoint 和纯推理完全不变。启用后：

1. 从指定 DiT block 取 `[B,T*H*W,D]` token，并对空间维平均，得到逐 latent 帧的 `z_t`。
2. 将视频采样后的 14 维动作再次按 latent 时间轴采样。对 93/121 这类 `4N+1` 视频和 4 倍因果 VAE，索引分别精确对应 `0,4,8,...`，首帧始终保留。
3. 轻量 Action Encoder 用瓶颈 MLP 将 `a_t` 映射到 `d_model`。
4. `Action Prediction Loss`：由 `z_t` 预测归一化动作，计算 MSE。
5. `Action Alignment Loss`：计算投影后的 `z_t` 与 Action Encoder 输出的逐时刻 cosine distance。

总损失：

```text
L = L_diffusion
  + sam3d_repa_weight * L_repa
  + action_loss_weight * (L_action_mse + action_alignment_weight * L_action_align)
```

动作只作为监督目标，不注入生成条件，避免训练时泄漏未来动作。

## 审计和生成归一化统计

先运行全量审计。该命令同时检查四段动作形状、14 维拼接顺序、
`joint_action/vector` 一致性、NaN/Inf、帧数范围和 FlowWAM RGB 字段：

```bash
python projects/sam3d/scripts/audit_worldarena_action_hdf5.py \
  /datassd/morka/cosmos-work/world-arena-ours-a5b076c7/dataset_track1 \
  --expected-count 1000 \
  --write-norm /datassd/morka/cosmos-work/world-arena-ours-a5b076c7/action_norm_stats.npz \
  --json-report /datassd/morka/cosmos-work/world-arena-ours-a5b076c7/action_h5_audit.json
```

生成的 `mean/std` 与 FlowWAM 一样采用全数据 z-score。训练时应固定使用同一份 `.npz`，不要在不同 worker 上各算一份。

## 启动动作监督训练

在原 `projects/sam3d/train_v93.sh` 环境变量基础上增加：

```bash
export ACTION_HDF5_ROOT=/path/to/paired_worldarena_h5_root
export ACTION_NORM_PATH=/path/to/action_norm_stats.npz
export ACTION_LOSS_WEIGHT=1.0
export ACTION_ALIGNMENT_WEIGHT=0.1
export ACTION_FEATURE_LAYER=7
export ACTION_HIDDEN_DIM=512
export ACTION_ALIGNMENT_OFFSET=0
```

然后照常在各节点运行 `projects/sam3d/train_v93.sh NODE_RANK`。当
`ACTION_LOSS_WEIGHT` 非零时，启动脚本会强制检查 H5 根目录和归一化文件，dataloader 也会强制每条视频存在可解析的动作 sidecar；配对不完整会直接报错，不会悄悄退化成无动作训练。

## 代码入口

- H5 schema、索引、归一化与时间采样：
  `cosmos_predict2/_src/predict2/datasets/local_datasets/worldarena_action_hdf5.py`
- 视频与动作同时间戳加载：
  `cosmos_predict2/_src/predict2/datasets/local_datasets/sam3d_video_dataset.py`
- Action Encoder、预测头和逐时刻 loss：
  `cosmos_predict2/_src/predict2/sam3d/action_alignment.py`
- 训练总损失接线：
  `cosmos_predict2/_src/predict2/models/sam3d_video2world_model.py`
