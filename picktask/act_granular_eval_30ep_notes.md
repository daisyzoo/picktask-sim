# ACT 细粒度分层评估记录（30 episodes）

日期：2026-07-09  
项目：MuJoCo `pickcup` 抓杯任务  
评估脚本：`picktask/scripts/pickcup_act_granular_eval.py`  
原始数据：`picktask/outputs/act_granular_eval_30ep.json`

## 1. 背景

ACT 从 `step_030000` 续训至 `step_080000` 后，标准闭环 eval（`pickcup_sim_eval.py`，成功标准：有效闭合 + 抬起 >5cm + 保持 1.5s）显示：

- `30k / 50k`：0/20
- `70k / 80k`：3/20（15%），且 70k→80k 无进一步提升

为定位瓶颈环节，新增细粒度 eval 脚本，对 **030000 / 050000 / 070000 / 080000** 四个 checkpoint 在同一批场景上统计分层成功率。

## 2. 评估设置

```bash
conda activate mujoco_demo
cd /Users/ke/Documents/wm/mujoco

mkdir -p picktask/outputs
python picktask/scripts/pickcup_act_granular_eval.py \
  --steps 30000,50000,70000,80000 \
  --episodes 30 \
  --seed 0 \
  --device cpu \
  --max-steps 3000 \
  --output-json picktask/outputs/act_granular_eval_30ep.json
```

| 参数 | 值 |
|------|-----|
| 数据集 stats | `picktask/pickcupdata/train/pickcup_train_200` |
| 场景来源 | `success-log`（`picktask/pickcup_train_200.log`，200 条成功场景） |
| episodes | 30（seed=0，四 checkpoint 共用同一批场景） |
| max_steps | 3000（sim dt 0.002s，约 6s） |
| device | cpu |

与标准 eval 的区别：**全程跑满 max_steps，不因达标提前结束**，便于统计各阶段通过率。

## 3. 分层指标定义

| 指标 | 字段 | 判定条件 |
|------|------|----------|
| 接触杯子 | `touch_cup` | 夹爪 geom 与杯体 geom 接触（dist ≤ 2mm） |
| 有效抓握 | `grasp` | 闭合时 `can_initiate_grasp`，或 `grasp.active` 物理吸附 |
| 抬起 ≥1cm | `lift_1cm` | episode 内 `max_cup_lift ≥ 0.01m` |
| 保持 ≥500ms | `hold_500ms` | 在 lift≥1cm 且 grasp 有效时，累计 hold ≥ 0.5s |
| 保持 ≥1.5s | `hold_1500ms` | 同上，累计 hold ≥ 1.5s |

标准 eval 成功标准更严：抬起 >**5cm** + 闭合有效接触 + 保持 1.5s + 终态仍有效。因此 `hold_1500ms`（1cm 门槛）通过率会高于标准 eval 的「全成功」率，但可用于观察相对趋势。

## 4. 评估结果（30 episodes，success-log，seed=0）

| Checkpoint | 接触杯子 | 有效抓握 | 抬起≥1cm | 保持≥500ms | 保持≥1.5s |
|------------|---------|---------|---------|-----------|----------|
| **step_030000** | 30/30 (**100%**) | 4/30 (**13%**) | 15/30 (**50%**) | 1/30 (**3%**) | 0/30 (**0%**) |
| **step_050000** | 30/30 (**100%**) | 3/30 (**10%**) | 13/30 (**43%**) | 2/30 (**7%**) | 2/30 (**7%**) |
| **step_070000** | 30/30 (**100%**) | 7/30 (**23%**) | 13/30 (**43%**) | 6/30 (**20%**) | 6/30 (**20%**) |
| **step_080000** | 30/30 (**100%**) | 10/30 (**33%**) | 16/30 (**53%**) | 6/30 (**20%**) | 5/30 (**17%**) |

各 checkpoint 耗时（脚本内计时）：30k ≈351s，50k ≈360s，70k ≈431s，80k ≈515s。

## 5. 与标准 eval（20 episodes）对照

标准 eval 使用 `pickcup_sim_eval.py`，成功标准见上（5cm + 1.5s 保持）。

| Checkpoint | 标准 eval（20 ep） | 细粒度 hold≥1.5s @1cm（30 ep） |
|------------|-------------------|--------------------------------|
| step_030000 | 0/20 = 0% | 0/30 = 0% |
| step_050000 | 0/20 = 0% | 2/30 = 7% |
| step_070000 | 3/20 = 15% | 6/30 = 20% |
| step_080000 | 3/20 = 15% | 5/30 = 17% |

趋势一致：70k 后进入平台期；80k 在抓握/抬升子指标上仍有小幅提升，但 **保持阶段（500ms / 1.5s）在 70k 后不再改善**。

## 6. 结论

### 6.1 能力分层

1. **接近能力已饱和**：四个 checkpoint 接触率均为 **100%**，说明策略已学会把夹爪送到杯子附近。
2. **主要瓶颈在抓握与保持**：有效抓握 10%–33%，保持 ≥1.5s 最高仅 **20%（70k）/ 17%（80k）**。
3. **抬升≠成功**：50% 的 episode 能抬过 1cm（30k），但几乎无法稳定保持；说明问题不在「够不到」，而在 **闭合质量与抓取后稳定性**。
4. **继续加步数收益递减**：70k→80k，抓握 23%→33%、抬≥1cm 43%→53% 有提升，但 hold≥500ms / hold≥1.5s 维持在 **20% / 17%**，与 loss 在 60k–80k 平台期（l1 ≈ 0.034）一致。

### 6.2 训练步数与跃迁区间

结合 20ep 标准 eval 与本次 30ep 细粒度 eval：

- **30k→50k**：标准 eval 仍 0%；50k ep10 曾出现 lift≈5.3cm + valid_close 但 hold 仅 0.14s（差 1.5s 门槛）。
- **50k→70k**：保持能力从 7% 跃升至 **20%**，标准 eval 首次达到 15%。
- **70k→80k**：子指标（抓握、抬升）略升，**保持率平台化**。

### 6.3 后续建议（与 step_030000 记录一致）

单纯续训步数意义不大，可优先尝试：

1. **resume optimizer**（当前续训为 fresh Adam，可能加剧平台期）
2. **降低 chunk_size**（100 对 ~173 帧/episode 偏长，闭环误差累积大）
3. **增加数据**或加入失败/边界样本
4. **调 eval 与训练对齐**：关注 hold 阶段行为，而非仅 approach/lift

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| `picktask/scripts/pickcup_act_granular_eval.py` | 细粒度 eval 脚本 |
| `picktask/scripts/pickcup_sim_eval.py` | 标准闭环 eval |
| `picktask/outputs/act_granular_eval_30ep.json` | 完整结果（含 30×4 条 per-episode 明细） |
| `picktask/outputs/act_granular_eval_30ep.log` | 运行日志（部分 run 因目录/重复进程不完整） |
| `picktask/step030000_eval_notes.md` | 早期 eval 排查与 20ep 对比记录 |
| `outputs/train/act_pickcup_train_200/checkpoints/step_*` | 各 checkpoint |

## 8. 运行备注

- 首次后台 run 因 `picktask/outputs/` 不存在导致 `tee` 失败；Python 进程仍完成评估并写出 JSON。
- 脚本已修复：写 JSON 前自动 `mkdir -p`；stdout 行缓冲，管道输出不空白。
- macOS 须先 `ACTPolicy.from_pretrained` 再 import mujoco（与 `pickcup_sim_eval.py` 相同）。
