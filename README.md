# picktask-sim

Unitree G1（23DOF）MuJoCo 抓取 / 场景仿真，以及 `qk_data` 校验脚本。

本仓库**只包含代码与文档**，不含大数据集与第三方完整依赖树。

## 包含内容

| 目录 | 说明 |
|---|---|
| `picktask/` | 桌面抓杯仿真（G1 + 键盘 teleop / 录制 / 训练入口） |
| `qk_validate/` | Pouring 等 LeRobot 数据在仿真中的回放与对比 |
| `home_scene/` | 厨房场景 stub（与 picktask 解耦） |
| `report/` | 汇报 HTML / 调研材料 |

## 不包含（需自行准备）

| 路径 | 说明 |
|---|---|
| `qk_data/` | 演示数据集（如 Pouring）；建议放 Hugging Face |
| `unitree_ros/` | 官方 G1 URDF（运行时需要） |
| `lerobot/` | LeRobot 源码（录制/训练可选） |

## 依赖安装（摘要）

1. **Conda 环境**（示例名 `mujoco_demo`）安装：`mujoco`、`numpy`、`pyarrow`、`opencv` 等（详见 `picktask/requirements-*.txt`）。
2. **G1 模型**：将 [unitree_ros](https://github.com/unitreerobotics/unitree_ros) 放到本仓库同级目录，使路径为：
   ```text
   <workspace>/unitree_ros/robots/g1_description/g1_23dof_mode_10.urdf
   ```
   （`picktask/lib/paths.py` 按工作区父目录解析。）
3. **macOS 3D Viewer**：使用 `mjpython` 启动带窗口脚本。

## 数据（Hugging Face）

Pouring 等大数据请单独托管。占位：

```text
HF dataset URL: <待上传后填写，例如 https://huggingface.co/datasets/<org>/qk-pouring>
本地放置: <workspace>/qk_data/Pouring
```

下载后目录结构需符合 LeRobot v3（含 `meta/info.json`、`data/`、`videos/`）。

## 快速运行

```bash
# 抓杯 teleop（macOS）
mjpython picktask/scripts/teleop.py

# Pouring 回放（左右对比）
mjpython qk_validate/replay_pouring.py --episode 0 --loop

# Pouring 对比（side + overlay）
mjpython qk_validate/replay_pouring_compare.py --episode 0 --loop --view both
```

## 许可与说明

仿真代码用于研究与评测链路演示；第三方模型 / 数据集遵循其各自许可证。
