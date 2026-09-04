#!/usr/bin/env python3
"""无头交互冒烟：灯/灶/柜门/手臂在三个工位上的状态机。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

import mujoco  # noqa: E402

from appliances import (  # noqa: E402
    cabinet_is_open,
    light_is_on,
    set_cabinet_door,
    set_stove_knob,
    sync_appliances,
    toggle_cabinet_door,
    toggle_light_switch,
    toggle_stove_knob,
)
from g1_mount import sync_g1_ctrl_to_qpos  # noqa: E402
from layout import LAYOUT_CHOICES, layout_from_name  # noqa: E402
from scene_composer import make_model_data  # noqa: E402
from teleop import resolve_key  # noqa: E402
import glfw  # noqa: E402


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_key_aliases() -> None:
    _check(resolve_key(ord("w")) == glfw.KEY_W, "w ASCII 别名")
    _check(resolve_key(ord("f")) == glfw.KEY_F, "f ASCII 别名")
    _check(resolve_key(ord("1")) == glfw.KEY_1, "1 ASCII 别名")
    _check(resolve_key(glfw.KEY_W) == glfw.KEY_W, "glfw W 直通")
    print("OK key aliases")


def test_layout(name: str) -> None:
    layout = layout_from_name(name)
    model, data = make_model_data(layout)
    sync_g1_ctrl_to_qpos(model, data)

    # 灯
    _check(light_is_on(model, data), f"{name}: 初始灯应亮")
    toggle_light_switch(model, data)
    sync_appliances(model, data)
    _check(not light_is_on(model, data), f"{name}: 关灯失败")
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "ceiling_lamp")
    _check(lid >= 0 and model.light_active[lid] == 0, f"{name}: ceiling 未熄")
    key_lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key_light")
    _check(key_lid >= 0 and float(model.light_diffuse[key_lid].sum()) < 0.3, f"{name}: 环境光未压暗")
    toggle_light_switch(model, data)
    sync_appliances(model, data)
    _check(light_is_on(model, data), f"{name}: 开灯失败")

    # 灶
    for i in range(4):
        on = toggle_stove_knob(model, data, i)
        sync_appliances(model, data)
        _check(on, f"{name}: 旋钮{i} 应开")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"stove_burner_{i}")
        _check(gid >= 0 and model.geom_rgba[gid][0] > 0.8, f"{name}: 炉盘{i} 未变橙")
        set_stove_knob(model, data, i, on=False)
        sync_appliances(model, data)

    # 柜门
    set_cabinet_door(model, data, open_=True)
    _check(cabinet_is_open(model, data), f"{name}: 开门失败")
    open_now = toggle_cabinet_door(model, data)
    _check(not open_now and not cabinet_is_open(model, data), f"{name}: 关门失败")

    # 手臂执行器
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_joint_pos")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint")
    data.ctrl[aid] = 0.5
    for _ in range(80):
        mujoco.mj_step(model, data)
        sync_appliances(model, data)
    ang = float(data.qpos[model.jnt_qposadr[jid]])
    _check(ang > 0.2, f"{name}: 肘关节未跟随 ctrl ({ang})")

    print(f"OK layout={name} ws={layout.workstation}")


def main() -> int:
    test_key_aliases()
    for name in LAYOUT_CHOICES:
        test_layout(name)
    print("ALL smoke_interact OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
