# Yellow Cup 资产

参考图 `1.jpg` / `2.jpg` / `3.jpg`，规格见 `information.txt`。

| 项目 | 数值 |
|------|------|
| 口部直径 | 8.3 cm |
| 底部直径 | 5.5 cm |
| 高度 | 10 cm |
| 容量 | 300 ml |
| 外色 | 闪电黄 |
| 内壁 | 不锈钢银白 |

## Phase 1：视觉光滑 + 碰撞圆柱

仿真中：
- **视觉**：程序生成截锥 OBJ（内外壁/底/口沿），缓存于 `picktask/assets/cache/cup_meshes/`
- **碰撞**：仍用隐藏的 `cup_*` 圆柱分段，抓取逻辑不变

## 用法

默认即为黄杯：

```bash
mjpython picktask/scripts/teleop.py
python picktask/scripts/pickcup_auto_demo.py --headless
```

切回红杯：

```bash
mjpython picktask/scripts/teleop.py --cup red
PICKCUP_CUP_VARIANT=red python picktask/scripts/pickcup_auto_demo.py --headless
```

## Phase 2：换商用 OBJ/STL（可选）

若之后有更高精度模型：覆盖 `assets/cache/cup_meshes/<variant>/cup_*.obj`，或改 `cup_geometry.ensure_cup_visual_meshes`；**碰撞仍建议保留圆柱分段**。
