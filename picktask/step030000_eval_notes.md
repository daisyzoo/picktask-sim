# ACT step_030000 评估与排查记录

日期：2026-07-08  
项目：MuJoCo `pickcup` 抓杯任务  
Checkpoint：`outputs/train/act_pickcup_train_200/checkpoints/step_030000`

## 1. 背景

ACT 训练从 `pickcup_train_200` 数据集开始，目标训练到 `80000` step。中途在 `step_030000` 保存 checkpoint 后，原训练进程被 `Ctrl-C` 中断。

本记录用于保留 `step_030000` 的中间评估结果，以及针对低成功率和 eval 崩溃问题的排查结论。

## 2. 数据与模型状态

训练数据：

- 数据集：`picktask/pickcupdata/train/pickcup_train_200`
- 成功 episode：200 条
- 总帧数：34672
- 平均 episode 长度：约 173 帧
- FPS：30
- 平均时长：约 5.8 秒

模型：

- checkpoint：`step_030000`
- ACT 配置：
  - `chunk_size = 100`
  - `n_action_steps = 100`
  - `n_obs_steps = 1`
  - 图像输入：`observation.images.head_camera`
  - proprio：8 维关节状态，不包含杯子位姿
  - action：8 维关节目标

训练日志在中断前最后记录：

```text
step 30000  loss=0.1161  l1=0.0908  kld=0.0025
```

续训已从 `step_030000` 重新启动，使用 fresh optimizer 继续训练到 `80000`。

## 3. 评估问题与修复

### 3.1 eval 脚本 segmentation fault

初次运行：

```bash
python picktask/scripts/pickcup_sim_eval.py \
  --policy outputs/train/act_pickcup_train_200/checkpoints/step_030000 \
  --dataset picktask/pickcupdata/train/pickcup_train_200 \
  --episodes 10 \
  --device cpu \
  --seed 0
```

出现 segmentation fault。

排查发现不是 `--device cpu` 未生效，而是 macOS 上的 import 顺序问题：

- 在 `ACTPolicy.from_pretrained()` 之前 import `mujoco`，可能触发 segfault。
- 在 `ACTPolicy.from_pretrained()` 之前 import `LeRobotDatasetMetadata`，也可能触发 segfault。

修复：

- 调整 `pickcup_sim_eval.py` 的 import 顺序：
  - 先 `bootstrap`
  - 再 `ACTPolicy.from_pretrained`
  - 再 import `LeRobotDatasetMetadata`
  - 最后 import `mujoco`

修复后 eval 可正常运行。

### 3.2 原默认 eval 时长过短

原默认参数：

```text
--max-steps 900
```

MuJoCo timestep 为 `0.002s`，因此 900 step 只覆盖约：

```text
900 * 0.002 = 1.8 秒
```

但训练 demo 平均约 5.8 秒，因此原 eval 会严重提前截断。

修复：

- `pickcup_sim_eval.py` 默认 `--max-steps` 改为 `3000`
- 约等于 6 秒，更接近训练 demo 平均时长

### 3.3 原随机 eval 分布不公平

训练集不是所有随机场景，而是 `pickcup_batch_record.py` 从随机场景中筛选出的 scripted 成功场景。

录制过程里原始 scripted 成功率约 9.6%，也就是说大量随机场景本来 scripted 都抓不起来。

原 eval 直接调用：

```python
pc.sample_random_scene(rng)
```

这会把模型评估在更难、更宽的分布上。

修复：

- `pickcup_sim_eval.py` 新增 `--scene-source`
  - `auto`：默认，优先读取训练日志中的成功场景
  - `success-log`：必须使用成功日志
  - `random`：旧逻辑，原始随机场景
- 新增 `--scenes-log`
- 默认自动读取：

```text
picktask/pickcup_train_200.log
```

并从其中 `OK saved=... robot=(...) cup=(...)` 记录恢复训练分布内的成功场景。

## 4. step_030000 评估结果

### 4.1 原随机场景评估

设置：

```bash
--episodes 20
--device cpu
--seed 0
--scene-source random
```

结果：

```text
成功率: 0/20 = 0.0%
```

现象：

- 最大抬升约 1 cm
- `valid_close=False`
- 基本没有形成有效抓取

### 4.2 拉长 horizon 后的随机场景评估

设置：

```bash
--episodes 5
--max-steps 3000
--scene-source random
```

结果：

```text
成功率: 0/5 = 0.0%
```

说明低成功率不只是因为原 eval 时长过短。

### 4.3 使用录制成功场景的公平评估

新的默认评估方式：

```bash
python picktask/scripts/pickcup_sim_eval.py \
  --policy outputs/train/act_pickcup_train_200/checkpoints/step_030000 \
  --dataset picktask/pickcupdata/train/pickcup_train_200 \
  --episodes 20 \
  --device cpu \
  --seed 0
```

评估配置：

```text
scene source: success-log (picktask/pickcup_train_200.log, 200 scenes)
max steps: 3000
```

结果：

```text
成功率: 0/20 = 0.0%
```

现象：

- 最高 lift 约 3.2 cm，仍低于成功标准 5 cm。
- 少数 episode 出现 `valid_close=True`，但没有维持抓取。
- 主要失败模式：能接近杯子，偶尔碰到或短暂抓到，但闭合位置/姿态不稳，无法抬杯。

此前手动取前 10 个日志成功场景做快速对照时曾出现：

```text
1/10 = 10%
```

但在随机抽样的 20 个训练成功场景上仍为 0%，说明 `step_030000` 仅偶然在少数容易场景成功，整体尚不可用。

## 5. 策略行为分析

打印 `step_030000` 在线动作轨迹后观察到：

- 策略不是完全不动。
- 约 2 秒左右会开始闭合夹爪。
- 肩 pitch、肩 yaw、肘关节会沿专家轨迹方向变化。
- 但闭合时机和手爪位置偏差较大，导致杯子没有被稳定夹住。

典型趋势：

```text
tick 000: 接近初始姿态，夹爪打开
tick 060: 夹爪已闭合，但 lift 仍为 0
tick 100+: 手臂继续上抬，但没有带起杯子
```

离线训练帧动作拟合误差：

```text
offline train-frame action MAE mean: 0.0151
right_shoulder_pitch_joint MAE: 0.0693
right_elbow_joint          MAE: 0.0337
gripper joints             MAE: ~0.0002
```

结论：

- 离线单帧动作拟合并非完全失败。
- 闭环接触任务对早期位置误差和闭合时机极敏感。
- 轻微误差在线滚动后会变成抓空或抓不稳。

## 6. 可能原因

综合判断，低成功率主要来自以下因素：

1. **训练步数不足**
   - `step_030000` 还处在中间阶段。
   - 模型已学到粗略轨迹，但未学稳接触与闭合细节。

2. **训练已在 30000 step 中断**
   - 原目标是 80000 step。
   - 当前 30k checkpoint 不能代表最终模型。

3. **ACT action queue 较长**
   - `n_action_steps=100`，对应约 3.3 秒动作 chunk。
   - 在线误差会在较长开环片段内累积。
   - 该点尚未完全定量验证，因为强制每步重新预测的诊断实验在 CPU 上太慢，被停止。

4. **数据量和观测难度**
   - 只有 200 条成功 demo。
   - 场景位置有随机变化。
   - proprio 不包含杯子位姿，模型必须从图像中定位杯子。
   - 对抓取任务而言 200 条数据偏少。

5. **成功判定严格**
   - 要求有效接触、抬起超过 5 cm，并维持 1.5 秒。
   - 短暂碰到或短暂夹住都不算成功。

## 7. 已完成修复

### 7.1 `pickcup_sim_eval.py`

已修复：

- macOS segfault 的 import 顺序
- 默认 horizon 从 900 改为 3000
- 新增训练成功场景分布评估：
  - `--scene-source auto`
  - `--scene-source success-log`
  - `--scene-source random`
  - `--scenes-log`

### 7.2 `pickcup_train_act.py`

已新增续训参数：

```bash
--resume-policy
--start-step
```

从 `step_030000` 继续训练命令：

```bash
python picktask/scripts/pickcup_train_act.py \
  --dataset picktask/pickcupdata/train/pickcup_train_200 \
  --repo-id local/pickcup_train_200 \
  --output outputs/train/act_pickcup_train_200 \
  --resume-policy outputs/train/act_pickcup_train_200/checkpoints/step_030000 \
  --steps 80000 \
  --device cpu \
  --batch-size 2 \
  --save-freq 10000
```

当前续训进程已启动，并使用：

```text
resume: step_030000 (start_step=30000, optimizer=fresh)
```

## 8. 后续建议

1. **继续训练到至少 60000 / 80000 step**
   - 每到 `step_040000`、`step_050000`、`step_060000`、`step_080000` 进行同一套公平评估。

2. **优先用训练成功场景评估**
   - 命令：

   ```bash
   python picktask/scripts/pickcup_sim_eval.py \
     --policy outputs/train/act_pickcup_train_200/checkpoints/step_040000 \
     --dataset picktask/pickcupdata/train/pickcup_train_200 \
     --episodes 20 \
     --device cpu \
     --seed 0
   ```

3. **随机场景评估作为更难泛化指标**
   - 命令：

   ```bash
   python picktask/scripts/pickcup_sim_eval.py \
     --policy outputs/train/act_pickcup_train_200/checkpoints/step_040000 \
     --dataset picktask/pickcupdata/train/pickcup_train_200 \
     --episodes 20 \
     --device cpu \
     --seed 0 \
     --scene-source random
   ```

4. **如果 80k 后仍低**
   - 考虑减小 ACT `n_action_steps`，例如 10 或 20。
   - 增加成功 demo 数量。
   - 增加杯子位姿或更强视觉特征。
   - 分阶段训练或改用 residual RL 做后续修正。

## 9. 当前结论

`step_030000` checkpoint 并非完全失败，但还未具备稳定抓杯能力。

它已经学到大致伸手和闭合趋势，离线动作拟合也不差；但在线闭环中，抓取接触位置和闭合时机不够准确，导致无法稳定抬杯。

因此当前 0% 成功率主要应理解为：

```text
30k 中间 checkpoint 尚未收敛 + 原 eval 分布/时长已修正后仍显示抓取能力不足
```

下一步应继续训练并在后续 checkpoint 上复测。
