"""厨房可交互家电/灯光运行时逻辑（Phase B）。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

LIGHT_SWITCH_JOINT = "light_switch_slide"
CEILING_LAMP_LIGHT = "ceiling_lamp"
ENV_LIGHTS = ("key_light", "fill_light", "living_lamp")
STOVE_KNOB_JOINTS = tuple(f"stove_knob_{i}_hinge" for i in range(4))
STOVE_BURNER_GEOMS = tuple(f"stove_burner_{i}" for i in range(4))
CABINET_HINGE_JOINT = "cabinet_stub_hinge"

SWITCH_ON_THRESHOLD = 0.008  # m
KNOB_ON_THRESHOLD = 0.35  # rad
CABINET_OPEN_ANGLE = 0.9
CABINET_CLOSED_ANGLE = 0.02
DEFAULT_LIGHT_FADE_SPEED = 2.8  # level / s（约 0.35s 完成）

_LIGHT_ON = np.array([0.95, 0.92, 0.85], dtype=np.float64)
_LIGHT_OFF = np.array([0.05, 0.05, 0.06], dtype=np.float64)
_ENV_ON = {
    "key_light": np.array([0.35, 0.35, 0.34], dtype=np.float64),
    "fill_light": np.array([0.12, 0.12, 0.14], dtype=np.float64),
    "living_lamp": np.array([0.55, 0.52, 0.45], dtype=np.float64),
}
_ENV_OFF_SCALE = 0.18
_BURNER_OFF = np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float64)
_BURNER_ON = np.array([0.95, 0.35, 0.05, 1.0], dtype=np.float64)


def _jid(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))


def _lid(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name))


def _gid(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def light_is_on(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    jid = _jid(model, LIGHT_SWITCH_JOINT)
    if jid < 0:
        return True
    return float(data.qpos[model.jnt_qposadr[jid]]) >= SWITCH_ON_THRESHOLD


def set_light_switch(
    model: mujoco.MjModel, data: mujoco.MjData, *, on: bool
) -> None:
    jid = _jid(model, LIGHT_SWITCH_JOINT)
    if jid < 0:
        return
    low, high = model.jnt_range[jid]
    data.qpos[model.jnt_qposadr[jid]] = float(high if on else low)
    data.qvel[model.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(model, data)


def toggle_light_switch(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    on = not light_is_on(model, data)
    set_light_switch(model, data, on=on)
    return on


def set_stove_knob(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: int,
    *,
    on: bool,
) -> None:
    if index < 0 or index >= len(STOVE_KNOB_JOINTS):
        return
    jid = _jid(model, STOVE_KNOB_JOINTS[index])
    if jid < 0:
        return
    low, high = model.jnt_range[jid]
    data.qpos[model.jnt_qposadr[jid]] = float(high if on else low)
    data.qvel[model.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(model, data)


def toggle_stove_knob(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> bool:
    jid = _jid(model, STOVE_KNOB_JOINTS[index])
    if jid < 0:
        return False
    cur = float(data.qpos[model.jnt_qposadr[jid]])
    on = cur < KNOB_ON_THRESHOLD
    set_stove_knob(model, data, index, on=on)
    return on


def cabinet_is_open(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    jid = _jid(model, CABINET_HINGE_JOINT)
    if jid < 0:
        return False
    return float(data.qpos[model.jnt_qposadr[jid]]) > 0.2


def set_cabinet_door(
    model: mujoco.MjModel, data: mujoco.MjData, *, open_: bool
) -> None:
    jid = _jid(model, CABINET_HINGE_JOINT)
    if jid < 0:
        return
    low, high = model.jnt_range[jid]
    target = CABINET_OPEN_ANGLE if open_ else CABINET_CLOSED_ANGLE
    data.qpos[model.jnt_qposadr[jid]] = float(np.clip(target, low, high))
    data.qvel[model.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(model, data)


def toggle_cabinet_door(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    open_now = not cabinet_is_open(model, data)
    set_cabinet_door(model, data, open_=open_now)
    return open_now


@dataclass
class LightFader:
    """灯光亮度 0..1 平滑过渡；开关关节在目标到达端点时对齐。"""

    level: float = 1.0
    target: float = 1.0
    speed: float = DEFAULT_LIGHT_FADE_SPEED

    def request(self, *, on: bool) -> None:
        self.target = 1.0 if on else 0.0

    def snap(self, *, on: bool) -> None:
        self.level = 1.0 if on else 0.0
        self.target = self.level

    def step(self, dt: float) -> float:
        if abs(self.target - self.level) < 1e-4:
            self.level = self.target
            return self.level
        step = self.speed * dt
        if self.target > self.level:
            self.level = min(self.target, self.level + step)
        else:
            self.level = max(self.target, self.level - step)
        return self.level


def apply_light_level(model: mujoco.MjModel, level: float) -> None:
    """按 0..1 插值吊灯与环境光（吊灯在接近熄灭时关闭）。"""
    level = float(np.clip(level, 0.0, 1.0))
    lid = _lid(model, CEILING_LAMP_LIGHT)
    if lid >= 0:
        model.light_active[lid] = 1 if level > 0.05 else 0
        model.light_diffuse[lid] = _LIGHT_OFF + level * (_LIGHT_ON - _LIGHT_OFF)

    for name in ENV_LIGHTS:
        elid = _lid(model, name)
        if elid < 0:
            continue
        base = _ENV_ON.get(name)
        if base is None:
            continue
        dim = base * _ENV_OFF_SCALE
        model.light_active[elid] = 1
        model.light_diffuse[elid] = dim + level * (base - dim)


def sync_appliances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    light_level: float | None = None,
) -> None:
    """每步更新灯光与炉盘外观。

    light_level 为 None 时按开关关节二值；演示可传入 0..1 做淡变。
    """
    if light_level is None:
        on = light_is_on(model, data)
        apply_light_level(model, 1.0 if on else 0.0)
    else:
        apply_light_level(model, light_level)

    for i, gname in enumerate(STOVE_BURNER_GEOMS):
        gid = _gid(model, gname)
        if gid < 0:
            continue
        jid = _jid(model, STOVE_KNOB_JOINTS[i])
        lit = False
        if jid >= 0:
            lit = float(data.qpos[model.jnt_qposadr[jid]]) >= KNOB_ON_THRESHOLD
        model.geom_rgba[gid] = _BURNER_ON if lit else _BURNER_OFF
