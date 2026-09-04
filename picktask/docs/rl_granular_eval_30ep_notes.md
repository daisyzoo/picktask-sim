# Residual RL 细粒度分层评估记录（30 episodes）

日期：2026-07-15  
项目：MuJoCo `pickcup` 抓杯任务  
评估脚本：`picktask/scripts/pickcup_rl_granular_eval.py`  
原始数据：`picktask/outputs/rl_granular_eval_30ep.json`  
运行日志：`picktask/outputs/rl_granular_eval_30ep.log`

## 1. 背景

基于 ACT `step_080000` 的 residual PPO（`outputs/rl/residual_act_80k`，10 万 env step）训练完成后，标准 RL eval 成功率为 **0%**。  
为定位残差策略相对纯 ACT 坏在哪一层，按 `act_granular_eval_30ep_notes.md` 同协议做细粒度评估，并增加更详细字段。

对照 ACT 细粒度（同 30 ep / success-log / seed=0）：`step_080000` 接触 100%、抓握 33%、hold≥1.5s 17%、标准成功约 13–15%。

## 2. 评估设置

```bash
conda activate mujoco_demo
cd /Users/ke/Documents/wm/mujoco
export PYTHONPATH="$(pwd)/lerobot/src:$PYTHONPATH"

python picktask/scripts/pickcup_rl_granular_eval.py \
  --act-policy outputs/train/act_pickcup_train_200/checkpoints/step_080000 \
  --act-dataset picktask/pickcupdata/train/pickcup_train_200 \
  --rl-dir outputs/rl/residual_act_80k \
  --policies act_only,residual_best,residual_final,residual_000095 \
  --episodes 30 \
  --seed 0 \
  --device auto \
  --max-steps 3000 \
  --verbose \
  --output-json picktask/outputs/rl_granular_eval_30ep.json
```

| 参数 | 值 |
|------|-----|
| ACT | `checkpoints/step_080000` |
| 场景 | success-log（`pickcup_train_200.log`，与 ACT granular 同 seed=0 抽 30 条） |
| max_steps | 3000 sim steps（约 6s，与 ACT granular 一致） |
| 残差缩放 | `DEFAULT_DELTA_SCALES * 0.5`（与训练 env 一致） |
| 策略 | `act_only`(Δ=0) / `best`(~1万步) / `final`(10万步) / `000095`(约中段) |

全程跑满 max_steps，不因达标提前结束。

## 3. 分层指标（详细版）

| 指标 | 判定 |
|------|------|
| 接触杯子 | 夹爪–杯 geom 接触 |
| 有效抓握 | 闭合可 initiate，或 grasp.active |
| 抬起 ≥1cm / ≥5cm | episode 内 max lift |
| 保持 ≥500ms / ≥1.5s | 在 lift≥1cm 且抓取有效时累计 hold |
| **标准成功** | 与 `check_episode_success` 一致（≥5cm + 有效闭合 + hold≥1.5s） |
| mean_\|Δ\| 等 | residual 动作幅度（act_only=0） |

## 4. 结果（30 episodes）

| Policy | 接触 | 抓握 | 抬≥1cm | 抬≥5cm | hold≥500ms | hold≥1.5s | **标准成功** | mean_\|Δ\| |
|--------|------|------|--------|--------|------------|-----------|--------------|-----------|
| **act_only** | 30/30 (**100%**) | 10/30 (**33%**) | 16/30 (**53%**) | 4/30 (**13%**) | 6/30 (**20%**) | 5/30 (**17%**) | **4/30 (13%)** | 0.000 |
| **residual_best** | 30/30 (100%) | 8/30 (27%) | 16/30 (53%) | 3/30 (10%) | 4/30 (13%) | **0/30 (0%)** | **0/30 (0%)** | 0.172 |
| **residual_final** | 30/30 (100%) | 11/30 (37%) | 20/30 (67%) | 3/30 (10%) | 5/30 (17%) | **0/30 (0%)** | **0/30 (0%)** | **0.355** |
| **residual_000095** | 30/30 (100%) | 10/30 (33%) | 18/30 (60%) | 4/30 (13%) | 6/30 (20%) | 2/30 (7%) | **1/30 (3%)** | 0.265 |

附加均值：

| Policy | mean_lift | mean_hold@1cm | 耗时 |
|--------|-----------|---------------|------|
| act_only | 0.0209 m | **0.601 s** | 451 s |
| residual_best | 0.0175 m | 0.143 s | 440 s |
| residual_final | 0.0189 m | 0.161 s | 462 s |
| residual_000095 | 0.0199 m | 0.367 s | 465 s |

`act_only` 与历史 ACT 80k granular（接触 100% / 抓握 33% / hold≥1.5s 17%）一致，说明本脚本协议对齐正确。

## 5. 结论：为什么成功率为 0？

### 5.1 Residual **没有学到有用修正，反而破坏保持**

1. **接近能力仍在**：所有策略接触 100% —— 残差没有毁掉 ACT 的接近。
2. **真正崩在 hold**：act_only 有 17% 能 hold≥1.5s、13% 标准成功；residual best/final 的 hold≥1.5s 与标准成功均为 **0%**。
3. **残差幅度越大越差**：`mean_|Δ|`：best 0.17 → mid 0.27 → final **0.36**，hold 均值从 0.60s（ACT）掉到 0.14–0.16s。
4. **final 略抬高 lift≥1cm（67%）但换不来成功**：短抬升↑、长保持↓ —— 典型「乱动夹爪 / 破坏闭合稳定性」。
5. **中段 000095 略好于 final**（标准成功 3%），说明后期 PPO 继续推大残差，属于 **过训 / 奖励错位**，不是步数不够。

### 5.2 因此：不要加长到 15–20 万

当前信号是 **残差在伤害 ACT**，加长只会更大 `|Δ|`。应改配方，而不是加步数。

## 6. 建议下一步（按优先级）

1. **先用纯 ACT 80k** 做业务基线；residual best/final **不要上线**。
2. **收紧残差**：训练时 residual_scales 再降 2–5×，或对夹爪维度几乎锁死（只允许臂小修正）。
3. **改 reward**：加大「掉落 / 失去 grasp / hold 中断」惩罚；降低纯 lift_delta 权重（避免为抬一点点而抖）。
4. **降探索**：更小 `log_std` / 更低 `ent_coef`，或 residual 输出 tanh 后再乘更小 scale。
5. **短训重跑**：例如 2–3 万步 + 更密 eval（每 2k），一旦 hold 掉就停；不要默认 10 万。
6. **数据侧**（中长期）：补「闭合–抬升–保持」边界 demo，或 ACT 侧降 chunk / 加 hold 相关监督 —— residual 救不了 ACT 本身 13% 的天花板，只能微调。

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| `picktask/scripts/pickcup_rl_granular_eval.py` | 本次详细细粒度脚本 |
| `picktask/outputs/rl_granular_eval_30ep.json` | 完整 per-episode 明细 |
| `picktask/act_granular_eval_30ep_notes.md` | ACT 对照笔记 |
| `outputs/rl/residual_act_80k/` | residual 权重 |

## 8. 一句话

**0% 不是「还没训够」，而是 residual 在接触仍满的情况下把 ACT 原本 ~13% 的保持/成功抹掉了；优先缩残差、改 reward、短训，而不是加到 15–20 万步。**
