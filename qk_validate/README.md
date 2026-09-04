# qk_validate

`qk_data` 数据在现有仿真里的校验 / 演示脚本。**只读复用** `picktask`，不修改其源码。

## 布局

| 路径 | 说明 |
|---|---|
| `qk_data/Pouring` | LeRobot v3 倒水数据（6D：腰+右臂） |
| `qk_validate/` | 本目录：校验与回放入口 |
| `picktask/` | 既有抓杯仿真（G1 23DOF mode_10），只作依赖 |

## Pouring 回放

桌前正对杯子；3D Viewer + 机器人视角。

### 基础版（左右并排）

```bash
mjpython qk_validate/replay_pouring.py --episode 0 --loop
python qk_validate/replay_pouring.py --headless --max-frames 30
```

### 对比版（增量：side + overlay）

三栏默认：`sim | real | overlay`，可切换；底部为 6D 关节 `|err|` 条。

```bash
mjpython qk_validate/replay_pouring_compare.py --episode 0 --loop
mjpython qk_validate/replay_pouring_compare.py --view both --alpha 0.45
python qk_validate/replay_pouring_compare.py --headless --max-frames 20
```

| 键 | 作用 |
|---|---|
| `1` / `2` / `3` | side / overlay / both |
| `[` / `]` | 降低 / 提高叠图 α（1=全仿真） |
| `SPACE` / `R` / `ESC` | 暂停 / 重播 / 退出 |

常用参数：`--episode N` · `--view side\|overlay\|both` · `--alpha 0.45` · `--no-error` · `--no-data-cam`

机型对齐：数据 `Unitree_G1_23Dof_RightArm5_Waist1` ↔ 仿真 `g1_23dof_mode_10`。
