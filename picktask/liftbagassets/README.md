# Lift Bag 资产与场景说明

Phase 1：**几何体近似**黑色方凳 + 托特包（刚体把手），旁置于现有桌+杯场景。

## 尺寸来源

见本目录 ` information.txt`（文件名含前导空格）与参考图 `stool.jpg` / `bag1.jpg` / `bag2.jpg`。

| 物体 | 尺寸 |
|------|------|
| 方凳 | 高 48cm，座面 28×28cm、厚 5cm；保留约 2.8cm 中央孔；与桌沿净空 1m（+X 侧） |
| 包体 | 横向 32cm、高 28cm、厚 14cm；前后包面从底部开始沿整段 smoothstep 曲线平滑收拢至 1cm 顶部开口，横向宽度不收缩；前后各保留一条扁平包口边，不设中央闭合条；封闭 mesh 仅用于质量与碰撞 |
| 带子 | 前后两组 1.2cm 宽、1.5mm 厚的连续平滑扁平带，根部固定片位于包体上方 5cm 内；左右间距按实测低/中/高位置 17.5/15.5/8cm 收窄，顶部两组带自然贴合；不可见分段 box 仅用于抓取碰撞 |

## 启动

在仓库根目录 `mujoco/`：

```bash
conda activate mujoco_demo
PICKCUP_HEAD_CAMERA_PREVIEW=0 mjpython picktask/scripts/liftbag_teleop.py
```

应看到：桌子+杯子（默认黄杯，`--cup red` 可切红杯）、旁侧黑凳、凳上包与双把手。`F`/`H` 开合夹爪，夹住把手可吸附拎起；抬升 ≥12cm 并保持 ≥1.5s 打印成功。

录制输出：`picktask/liftbagdata/sessions/liftbag_YYYYMMDD_HHMMSS/`。

## Phase 2：换 OBJ/STL 外观（可选）

当前无 mesh。若之后有模型：

1. 将 `.obj`/`.stl` 放到本目录（单位米、Z 向上）。
2. 在 MJCF 中用 `<mesh>` + 视觉 geom 引用；**碰撞仍建议保留 box/capsule**。
3. 把手可继续用分段扁平刚体带，直到需要软带仿真。

可从 Sketchfab / Objaverse / BlenderKit 等下载可商用授权的 stool、tote bag。
