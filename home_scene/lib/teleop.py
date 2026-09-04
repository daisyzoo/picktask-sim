"""home_scene 键盘遥操作：通过 23 个 position 执行器驱动 G1。

不依赖 picktask。macOS 请用 mjpython 启动入口脚本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import glfw
import mujoco
import numpy as np

from g1_mount import ACTUATED_JOINTS, G1_JOINTS

JOINT_STEP = 0.08

# 右臂 + 腰：常用操作
RIGHT_ARM_BINDINGS: dict[int, tuple[str, float]] = {
    glfw.KEY_W: ("waist_yaw_joint", JOINT_STEP),
    glfw.KEY_S: ("waist_yaw_joint", -JOINT_STEP),
    glfw.KEY_T: ("right_shoulder_pitch_joint", JOINT_STEP),
    glfw.KEY_G: ("right_shoulder_pitch_joint", -JOINT_STEP),
    glfw.KEY_A: ("right_shoulder_roll_joint", JOINT_STEP),
    glfw.KEY_D: ("right_shoulder_roll_joint", -JOINT_STEP),
    glfw.KEY_Z: ("right_shoulder_yaw_joint", JOINT_STEP),
    glfw.KEY_X: ("right_shoulder_yaw_joint", -JOINT_STEP),
    glfw.KEY_C: ("right_elbow_joint", -JOINT_STEP),
    glfw.KEY_V: ("right_elbow_joint", JOINT_STEP),
    glfw.KEY_B: ("right_wrist_roll_joint", JOINT_STEP),
    glfw.KEY_N: ("right_wrist_roll_joint", -JOINT_STEP),
}

# 左臂
LEFT_ARM_BINDINGS: dict[int, tuple[str, float]] = {
    glfw.KEY_I: ("left_shoulder_pitch_joint", JOINT_STEP),
    glfw.KEY_K: ("left_shoulder_pitch_joint", -JOINT_STEP),
    glfw.KEY_J: ("left_shoulder_roll_joint", JOINT_STEP),
    glfw.KEY_L: ("left_shoulder_roll_joint", -JOINT_STEP),
    glfw.KEY_U: ("left_shoulder_yaw_joint", JOINT_STEP),
    glfw.KEY_O: ("left_shoulder_yaw_joint", -JOINT_STEP),
    glfw.KEY_Y: ("left_elbow_joint", -JOINT_STEP),
    glfw.KEY_H: ("left_elbow_joint", JOINT_STEP),
    glfw.KEY_COMMA: ("left_wrist_roll_joint", JOINT_STEP),
    glfw.KEY_PERIOD: ("left_wrist_roll_joint", -JOINT_STEP),
}

DIRECT_BINDINGS = {**RIGHT_ARM_BINDINGS, **LEFT_ARM_BINDINGS}

# MuJoCo 在部分平台会传入 ASCII（ord('w')=119）而非 glfw.KEY_W(=87)
ASCII_KEY_ALIASES: dict[int, int] = {
    ord("w"): glfw.KEY_W,
    ord("W"): glfw.KEY_W,
    ord("s"): glfw.KEY_S,
    ord("S"): glfw.KEY_S,
    ord("a"): glfw.KEY_A,
    ord("A"): glfw.KEY_A,
    ord("d"): glfw.KEY_D,
    ord("D"): glfw.KEY_D,
    ord("z"): glfw.KEY_Z,
    ord("Z"): glfw.KEY_Z,
    ord("x"): glfw.KEY_X,
    ord("X"): glfw.KEY_X,
    ord("c"): glfw.KEY_C,
    ord("C"): glfw.KEY_C,
    ord("v"): glfw.KEY_V,
    ord("V"): glfw.KEY_V,
    ord("b"): glfw.KEY_B,
    ord("B"): glfw.KEY_B,
    ord("n"): glfw.KEY_N,
    ord("N"): glfw.KEY_N,
    ord("t"): glfw.KEY_T,
    ord("T"): glfw.KEY_T,
    ord("g"): glfw.KEY_G,
    ord("G"): glfw.KEY_G,
    ord("f"): glfw.KEY_F,
    ord("F"): glfw.KEY_F,
    ord("i"): glfw.KEY_I,
    ord("I"): glfw.KEY_I,
    ord("j"): glfw.KEY_J,
    ord("J"): glfw.KEY_J,
    ord("k"): glfw.KEY_K,
    ord("K"): glfw.KEY_K,
    ord("l"): glfw.KEY_L,
    ord("L"): glfw.KEY_L,
    ord("u"): glfw.KEY_U,
    ord("U"): glfw.KEY_U,
    ord("o"): glfw.KEY_O,
    ord("O"): glfw.KEY_O,
    ord("y"): glfw.KEY_Y,
    ord("Y"): glfw.KEY_Y,
    ord("h"): glfw.KEY_H,
    ord("H"): glfw.KEY_H,
    ord("r"): glfw.KEY_R,
    ord("R"): glfw.KEY_R,
    ord("p"): glfw.KEY_P,
    ord("P"): glfw.KEY_P,
    ord(","): glfw.KEY_COMMA,
    ord("."): glfw.KEY_PERIOD,
    ord("-"): glfw.KEY_MINUS,
    ord("="): glfw.KEY_EQUAL,
    ord("["): glfw.KEY_LEFT_BRACKET,
    ord("]"): glfw.KEY_RIGHT_BRACKET,
    ord("0"): glfw.KEY_0,
    ord("1"): glfw.KEY_1,
    ord("2"): glfw.KEY_2,
    ord("3"): glfw.KEY_3,
    ord("4"): glfw.KEY_4,
}


def resolve_key(key: int | str) -> int:
    if isinstance(key, str):
        if len(key) == 1:
            key = ord(key)
        else:
            return -1
    if isinstance(key, int) and key in ASCII_KEY_ALIASES:
        return ASCII_KEY_ALIASES[key]
    return int(key)


HELP_TEXT = """
键盘遥操作（G1 全 23DOF，pelvis 固定）
────────────────────────────────────
右臂/腰:
  W/S  腰 yaw ±
  T/G  右肩 pitch ±
  A/D  右肩 roll ±
  Z/X  右肩 yaw ±
  C/V  右肘 ±
  B/N  右腕 roll ±
左臂:
  I/K  左肩 pitch ±
  J/L  左肩 roll ±
  U/O  左肩 yaw ±
  Y/H  左肘 ±
  ,/.  左腕 roll ±
全关节微调（含腿）:
  [ / ]  选择上一个/下一个关节
  - / =  当前关节 − / +
  R      复位全部关节到 0
  P      打印当前关节角
家电/灯光/柜门:
  F      切换天花板灯（墙面开关；关灯时环境光也会变暗）
  1/2/3/4  切换灶台旋钮 0–3（炉盘变橙红）
  0      切换柜门开/关
  Esc    退出（关窗口也可）

提示: 先点击 MuJoCo 3D 窗口再按键；客厅工位也能控厨房灯/灶（整套公寓同场景）。
"""


def _joint_range(model: mujoco.MjModel, joint_name: str) -> tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return -3.14, 3.14
    low, high = model.jnt_range[jid]
    if low >= high:
        return -3.14, 3.14
    return float(low), float(high)


def _qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return float(data.qpos[model.jnt_qposadr[jid]])


@dataclass
class TeleopState:
    model: mujoco.MjModel
    data: mujoco.MjData
    ctrl: dict[str, float] = field(default_factory=dict)
    act_ids: dict[str, int] = field(default_factory=dict)
    select_idx: int = 12  # 默认选中腰
    quit: bool = False
    dirty_print: bool = False

    def __post_init__(self) -> None:
        for name in ACTUATED_JOINTS:
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos"
            )
            if aid < 0:
                raise RuntimeError(f"缺少执行器: {name}_pos")
            self.act_ids[name] = int(aid)
            self.ctrl[name] = _qpos(self.model, self.data, name)
        self.apply()

    def apply(self) -> None:
        for name, value in self.ctrl.items():
            low, high = _joint_range(self.model, name)
            clipped = float(np.clip(value, low, high))
            self.ctrl[name] = clipped
            self.data.ctrl[self.act_ids[name]] = clipped

    def nudge(self, joint_name: str, delta: float) -> None:
        if joint_name not in self.ctrl:
            return
        self.ctrl[joint_name] = self.ctrl[joint_name] + delta
        self.apply()

    def reset_pose(self) -> None:
        for name in ACTUATED_JOINTS:
            self.ctrl[name] = 0.0
        self.apply()

    def selected_joint(self) -> str:
        return G1_JOINTS[self.select_idx % len(G1_JOINTS)]

    def handle_key(self, key: int | str) -> None:
        key = resolve_key(key)
        if key < 0:
            return
        if key in (glfw.KEY_ESCAPE,):
            self.quit = True
            return
        if key == glfw.KEY_R:
            self.reset_pose()
            print("复位关节 → 0")
            return
        if key == glfw.KEY_P:
            self.dirty_print = True
            return
        if key == glfw.KEY_F:
            from appliances import sync_appliances, toggle_light_switch

            on = toggle_light_switch(self.model, self.data)
            sync_appliances(self.model, self.data)
            print(f"天花板灯 → {'ON' if on else 'OFF'}")
            return
        if key == glfw.KEY_0:
            from appliances import toggle_cabinet_door

            open_now = toggle_cabinet_door(self.model, self.data)
            print(f"柜门 → {'OPEN' if open_now else 'CLOSED'}")
            return
        if key in (
            glfw.KEY_1,
            glfw.KEY_2,
            glfw.KEY_3,
            glfw.KEY_4,
            glfw.KEY_KP_1,
            glfw.KEY_KP_2,
            glfw.KEY_KP_3,
            glfw.KEY_KP_4,
            glfw.KEY_KP_0,
        ):
            from appliances import sync_appliances, toggle_stove_knob

            idx_map = {
                glfw.KEY_1: 0,
                glfw.KEY_2: 1,
                glfw.KEY_3: 2,
                glfw.KEY_4: 3,
                glfw.KEY_KP_1: 0,
                glfw.KEY_KP_2: 1,
                glfw.KEY_KP_3: 2,
                glfw.KEY_KP_4: 3,
            }
            if key == glfw.KEY_KP_0:
                from appliances import toggle_cabinet_door

                open_now = toggle_cabinet_door(self.model, self.data)
                print(f"柜门 → {'OPEN' if open_now else 'CLOSED'}")
                return
            idx = idx_map[key]
            on = toggle_stove_knob(self.model, self.data, idx)
            sync_appliances(self.model, self.data)
            print(f"灶台旋钮{idx} → {'ON' if on else 'OFF'}")
            return
        if key == glfw.KEY_LEFT_BRACKET:
            self.select_idx = (self.select_idx - 1) % len(G1_JOINTS)
            print(f"选中关节 [{self.select_idx}] {self.selected_joint()}")
            return
        if key == glfw.KEY_RIGHT_BRACKET:
            self.select_idx = (self.select_idx + 1) % len(G1_JOINTS)
            print(f"选中关节 [{self.select_idx}] {self.selected_joint()}")
            return
        if key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.nudge(self.selected_joint(), -JOINT_STEP)
            return
        if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.nudge(self.selected_joint(), JOINT_STEP)
            return
        binding = DIRECT_BINDINGS.get(key)
        if binding is not None:
            joint, delta = binding
            self.nudge(joint, delta)

    def maybe_print(self) -> None:
        if not self.dirty_print:
            return
        self.dirty_print = False
        vals = " ".join(
            f"{n.split('_')[0][:1]}{_qpos(self.model, self.data, n):+.2f}"
            for n in ACTUATED_JOINTS
        )
        print(f"qpos: {vals}")


def make_key_callback(state: TeleopState):
    def _cb(key: int | str) -> None:
        state.handle_key(key)

    return _cb
