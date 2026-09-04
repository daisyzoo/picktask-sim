# Residual RL v2 细粒度分层评估记录（30 episodes）

日期：2026-07-16  
项目：MuJoCo `pickcup` 抓杯任务  
评估脚本：`picktask/scripts/pickcup_rl_granular_eval.py`（已支持 residual obs 8/12）  
原始数据：`picktask/outputs/rl_granular_eval_v2_30ep.json`  
运行日志：`picktask/outputs/rl_granular_eval_v2_30ep.log`

## 1. 背景

`outputs/rl/residual_act_v2` 短训已完成（约 3 万步量级；`best` 较早落盘，`final` 2026-07-15 15:08）。  
相对 v1（`residual_act_80k`）配方变更：观测 **12 维**、残差 scale≈DEFAULT×0.1、夹爪通道锁死、reward/探索收紧。

录取门槛（readme）：细粒度/标准成功 **不低于纯 ACT（约 13%）** 再保留。

## 2. 评估设置

```bash
conda activate mujoco_demo
cd /Users/ke/Documents/wm/mujoco
export PYTHONPATH="$(pwd)/lerobot/src:$PYTHONPATH"

python picktask/scripts/pickcup_rl_granular_eval.py \
  --act-policy outputs/train/act_pickcup_train_200/checkpoints/step_080000 \
  --act-dataset picktask/pickcupdata/train/pickcup_train_200 \
  --rl-dir outputs/rl/residual_act_v2 \
  --policies act_only,residual_best,residual_final \
  --episodes 30 \
  --seed 0 \
  --device auto \
  --max-steps 3000 \
  --verbose \
  --output-json picktask/outputs/rl_granular_eval_v2_30ep.json
```

| 参数 | 值 |
|------|-----|
| ACT | `checkpoints/step_080000` |
| 场景 | success-log（与 ACT / v1 granular 同 seed=0 抽 30 条） |
| max_steps | 3000 |
| 残差缩放 | `DEFAULT×0.1`，夹爪=0（与训练一致） |
| residual obs | **12 维**（proprio + gripper/contact/lift/grasp） |
| 策略 | `act_only` / `best` / `final` |

## 3. 结果（30 episodes）

| Policy | 接触 | 抓握 | 抬≥1cm | 抬≥5cm | hold≥500ms | hold≥1.5s | **标准成功** | mean_\|Δ\| |
|--------|------|------|--------|--------|------------|-----------|--------------|-----------|
| **act_only** | 30/30 (**100%**) | 10/30 (**33%**) | 16/30 (**53%**) | 4/30 (**13%**) | 6/30 (**20%**) | 5/30 (**17%**) | **4/30 (13%)** | 0.000 |
| **residual_best** | 30/30 (100%) | 9/30 (30%) | 15/30 (50%) | 2/30 (7%) | 4/30 (13%) | **0/30 (0%)** | **0/30 (0%)** | **0.0039** |
| **residual_final** | 30/30 (100%) | 9/30 (30%) | 17/30 (57%) | 3/30 (10%) | 6/30 (20%) | **0/30 (0%)** | **0/30 (0%)** | **0.017** |

附加均值：

| Policy | mean_lift | mean_hold@1cm | 耗时 |
|--------|-----------|---------------|------|
| act_only | 0.0209 m | **0.601 s** | 471 s |
| residual_best | 0.0167 m | 0.131 s | 439 s |
| residual_final | 0.0179 m | 0.189 s | 441 s |

`act_only` 与历史 ACT 80k / v1 granular 一致（接触 100% / 抓握 33% / 标准成功 13%），协议对齐正确。

## 4. 与 v1（residual_act_80k）对照

| | v1 best | v1 final | **v2 best** | **v2 final** |
|--|---------|----------|-------------|--------------|
| 标准成功 | 0% | 0% | **0%** | **0%** |
| hold≥1.5s | 0% | 0% | **0%** | **0%** |
| mean_\|Δ\| | 0.172 | 0.355 | **0.004** | **0.017** |
| mean_hold@1cm | 0.14 s | 0.16 s | 0.13 s | 0.19 s |

v2 **成功把残差幅度压下去一个数量级**，但 **hold / 标准成功仍被抹成 0%**——说明问题不只是「Δ 太大」，小残差在 hold 窗口仍足以破坏 ACT 的闭合稳定性。

## 5. 结论

1. **训练已完成**；`best`/`final` 均未达到录取门槛（≥ ACT 13%）。
2. **配方部分生效**：`|Δ|` 远小于 v1，夹爪锁死也生效；接触能力仍在（100%）。
3. **仍崩在 hold**：act_only 有 17% hold≥1.5s、13% 标准成功；v2 best/final 均为 **0%**。
4. **final 比 best `|Δ|` 更大**（0.004→0.017），hold 均值略回升但仍远低于 ACT——继续训没有越过门槛。
5. **不要加长训练**；也不要上线 residual_best/final。业务基线继续用 **纯 ACT 80k**。

## 6. 建议下一步（按优先级）

1. **停 residual-v2 加长**；保留本次 JSON/笔记作负结果归档。
2. 若仍要 residual：考虑 **只在接近/抓取阶段开残差、hold 阶段强制 Δ=0**（阶段门控），或把 hold 稳定性做成硬约束。
3. Reward 再压「扰动」：对 hold 中的 `action`/`Δ` 施加强惩罚；lift shaping 继续降权。
4. 中长期：补 hold 边界 demo / 加强 ACT 本身，比继续拧 residual 更有希望抬过 13%。

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| `outputs/rl/residual_act_v2/` | v2 权重（best/final + ckpt） |
| `picktask/outputs/rl_granular_eval_v2_30ep.json` | 完整 per-episode |
| `picktask/rl_granular_eval_30ep_notes.md` | v1 对照 |
| `picktask/scripts/pickcup_rl_granular_eval.py` | 本次修复：按 obs_dim 组装 8/12 维观测 |

## 8. 一句话

**v2 训完了；残差变小了，但标准成功仍是 0%（ACT 对照 13%）——未过录取门槛，继续用纯 ACT，不要加长 residual。**
