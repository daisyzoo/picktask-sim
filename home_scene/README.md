# home_scene

独立家庭场景：一套程序化公寓（厨房 + 客厅）、G1 全 23DOF、头显预览、可交互灯光/灶台。
与 `picktask` / `vla_sim_eval` 解耦。

## 边界

| 允许 | 禁止 |
|------|------|
| 只读 `unitree_ros/` | import picktask / vla_sim_eval |
| 本目录 assets/data | 写入 picktask 数据目录 |

## 框架

```text
HomeSpec（公寓）
  └─ RoomSpec（厨房 / 客厅）
       └─ Workstation（台前 / 柜前 / 茶几前 …）
            └─ SceneLayout（一次 episode：站哪个工位、柜门开多大）
```

- **场景不知任务**：Layout 只描述有什么、机器人站哪。
- **任务不建房间**：`home_close_cabinet` 只认铰链角。
- **移动 = 工位瞬移**：`pelvis` 固定在世界系，换 `--layout` 即换挂载点，暂不做行走。
- **资产可替换**：夹具先 stub，日后同 id 换成 RoboCasa mesh。

`pelvis` 是 G1 运动学树的根连杆（骨盆/髋）。固定 pelvis = 机器人在房间里的位置和朝向锁死，腿和手臂仍可动，但不会走路或摔倒。

## 快速开始

```bash
conda activate mujoco_demo
cd /Users/ke/Documents/wm/mujoco/home_scene
python scripts/smoke_build.py
python scripts/smoke_close_cabinet.py

# macOS 必须 mjpython
mjpython scripts/teleop.py                      # 厨房台前
mjpython scripts/teleop.py --layout close_cabinet
mjpython scripts/teleop.py --layout living      # 客厅茶几前
mjpython scripts/view_scene.py --layout living
```

## 当前内容

| 房间 | 尺寸 | 工位 | 内容 |
|------|------|------|------|
| kitchen | 6m×5m | `counter_front` `sink_front` `stove_front` `cabinet_west` | 台面（水槽/备餐/灶）、铰链柜、灯开关、南窗 |
| living | 4m×5m | `table_front` | 茶几、沙发 stub、落地灯；东墙门洞连通厨房 |

- **G1-23** 全关节 position 执行器（pelvis 固定）
- **头显 D435** 子进程 OpenCV 预览
- **可交互**：`F` 灯开关；`1–4` 灶台旋钮

## 自动演示 / 录制

头显长视频：厨房 → 黑场字幕 → 关柜 → 转场 → 客厅。
Demo 控制：只平滑驱动腰+双臂，腿 hold；灯光淡变；H.264 30fps。

```bash
mjpython scripts/auto_demo.py --record
python scripts/auto_demo.py --headless --record --no-realtime
python scripts/smoke_interact.py
```

## RoboCasa

资产软链到 `assets/robocasa/`。场景仍用程序化 stub。

```bash
conda activate robocasa
python scripts/fetch_robocasa_assets.py --link --scan
```
