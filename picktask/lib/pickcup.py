from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from head_camera_preview import HeadCameraPreviewProcess
from cup_geometry import ensure_cup_visual_meshes, profile_from_segments
from lerobot_recorder import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEFAULT_TASK,
    PickcupLeRobotRecorder,
    RECORD_FPS,
)
from paths import (
    DATA_DIR,
    G1_COMPILED_MJCF_CACHE,
    G1_DESCRIPTION_DIR,
    G1_URDF_PATH,
    SESSIONS_DATA_DIR,
    TRAIN_DATA_DIR,
    YELLOWCUP_ASSETS_DIR,
)


# ==================== 文件与场景配置 ====================
HEAD_CAMERA_NAME = "head_camera"
HEAD_CAMERA_WINDOW = "robot_view · D435i"
RECORD_PERIOD = 1.0 / RECORD_FPS

# 宇树真机 RealSense D435i（彩色流）标定与设备信息。
D435_MODEL = "Intel RealSense D435i"
D435_SERIAL = "233622077487"
D435_FIRMWARE = "5.16.0.1"
D435_FIRMWARE_RECOMMENDED = "5.17.0.10"

# G1-23 官方 URDF 中 d435_link 相对 torso_link 的安装外参。
# MuJoCo 相机沿局部 -Z 观察、+Y 向上，因此四元数同时包含 URDF
# pitch=0.8307767 rad 和 ROS 相机轴到 MuJoCo 轴的转换。
D435_TORSO_POSITION = (0.0576235, 0.01753, 0.42987)
D435_MUJOCO_QUATERNION = (
    0.65925248,
    0.25570719,
    -0.25570719,
    -0.65925248,
)
D435_COLOR_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)  # 640×480
# 真机实测完整像素内参（彩色流）。
D435_COLOR_FX = 605.5006
D435_COLOR_FY = 605.4136
D435_COLOR_CX = 326.1470
D435_COLOR_CY = 249.5715
D435_COLOR_FOCAL_PIXEL = (D435_COLOR_FX, D435_COLOR_FY)
# RealSense/OpenCV 主点为左上角原点；MuJoCo 要相对图像中心，且图像平面 +Y 向上。
D435_COLOR_PRINCIPAL_PIXEL = (
    D435_COLOR_CX - D435_COLOR_RESOLUTION[0] / 2,
    D435_COLOR_RESOLUTION[1] / 2 - D435_COLOR_CY,
)
# sensorsize 只决定像素到传感器长度单位的比例；6.4 x 4.8 保持 0.01 unit/pixel。
D435_COLOR_SENSOR_SIZE = (6.4, 4.8)
# 由 fx/fy 推导的 FOV（与真机实测 55.71° × 43.24° 一致）。
D435_COLOR_HFOV_DEG = math.degrees(
    2.0 * math.atan(D435_COLOR_RESOLUTION[0] / (2.0 * D435_COLOR_FX))
)
D435_COLOR_VFOV_DEG = math.degrees(
    2.0 * math.atan(D435_COLOR_RESOLUTION[1] / (2.0 * D435_COLOR_FY))
)

TABLE_LENGTH = 1.50
TABLE_WIDTH = 0.60
TABLE_HEIGHT = 0.75
TABLE_THICKNESS = 0.025
TABLE_SURFACE_Z = TABLE_HEIGHT + TABLE_THICKNESS / 2
FLOOR_Z = 0.0
CUP_MIN_WORLD_Z = TABLE_SURFACE_Z - 0.01

# 桌子：长边 1.5m 沿 X（y=±0.30 的两条边），短边 0.6m 沿 Y（x=±0.75 的两条边）。
# 机器人站在长边外侧（y 负侧），绕 Z 轴 +90° 后面朝 +Y，正对该 1.5m 长边。
# 桌边到机器人前脚/小腿碰撞 mesh 最前缘的最小水平净空。默认零位姿下，
# 下肢最前缘相对 pelvis 沿朝桌方向伸出 0.142363 m。额外保留 3 cm
# 碰撞裕量，避免默认右手 mesh 在桌沿处发生数毫米穿透。
ROBOT_TABLE_CLEARANCE = 0.15
ROBOT_TABLE_COLLISION_MARGIN = 0.03
ROBOT_LOWER_BODY_FORWARD_EXTENT = 0.14236302056320188
ROBOT_TABLE_STANDOFF = (
    ROBOT_TABLE_CLEARANCE
    + ROBOT_TABLE_COLLISION_MARGIN
    + ROBOT_LOWER_BODY_FORWARD_EXTENT
)
DEFAULT_ROBOT_YAW = math.pi / 2
DEFAULT_ROBOT_POSITION = (
    0.12,
    -(TABLE_WIDTH / 2 + ROBOT_TABLE_STANDOFF),
    0.793,
)
DEFAULT_CUP_POSITION = [
    0.12,
    -(TABLE_WIDTH / 2) + 0.08,
    TABLE_SURFACE_Z + 0.0025,
]
CUP_POSITION = DEFAULT_CUP_POSITION
ROBOT_POSITION = DEFAULT_ROBOT_POSITION

# 批量数据随机化默认范围（相对 DEFAULT_* 的偏移）。
ROBOT_X_RANGE = (-0.08, 0.08)
# 只允许从 15 cm 基准位置继续后移，不能随机向桌子靠近。
ROBOT_Y_RANGE = (-0.12, 0.0)
CUP_X_RANGE = (-0.08, 0.08)
CUP_Y_RANGE = (-0.12, 0.12)
CUP_YAW_RANGE = (-0.35, 0.35)

# 杯子外观变体：碰撞仍用圆柱分段（隐藏）；视觉用光滑截锥 mesh。
# yellow 参考 assets/yellowcupassets（底 Ø5.5cm / 口 Ø8.3cm / 高 10cm，闪电黄）。
CUP_VARIANTS: dict[str, dict[str, object]] = {
    "yellow": {
        "label": "闪电黄不锈钢随行杯",
        "assets_dir": YELLOWCUP_ASSETS_DIR,
        "materials": {
            "cup_outer": "0.95 0.78 0.12 1",
            "cup_inner": "0.72 0.74 0.76 1",
        },
        # 总质量约 0.11 kg（钢杯感）；分段半径由底 2.75cm 线性过渡到口 4.15cm。
        "segments": (
            ("cup_bottom", 0.0275, 0.005, "cup_inner", 0.020, 0.0025),
            ("cup_lower", 0.0303, 0.014, "cup_outer", 0.025, 0.020),
            ("cup_mid1", 0.0338, 0.014, "cup_outer", 0.025, 0.045),
            ("cup_mid2", 0.0373, 0.014, "cup_outer", 0.025, 0.070),
            ("cup_rim", 0.0415, 0.006, "cup_inner", 0.015, 0.096),
        ),
    },
    "red": {
        "label": "红色纸杯（几何近似）",
        "assets_dir": None,
        "materials": {
            "cup_outer": "0.86 0.12 0.15 1",
            "cup_inner": "0.98 0.98 0.96 1",
        },
        "segments": (
            ("cup_bottom", 0.030, 0.005, "cup_inner", 0.006, 0.0025),
            ("cup_lower", 0.032, 0.014, "cup_outer", 0.008, 0.020),
            ("cup_mid1", 0.034, 0.014, "cup_outer", 0.008, 0.045),
            ("cup_mid2", 0.036, 0.014, "cup_outer", 0.008, 0.070),
            ("cup_rim", 0.040, 0.006, "cup_inner", 0.004, 0.096),
        ),
    },
}
DEFAULT_CUP_VARIANT = "yellow"


def resolve_cup_variant(cli_value: str | None = None) -> str:
    """解析杯子变体：CLI > 环境变量 PICKCUP_CUP_VARIANT > 默认 yellow。"""
    raw = cli_value if cli_value is not None else os.environ.get("PICKCUP_CUP_VARIANT")
    variant = (raw or DEFAULT_CUP_VARIANT).strip().lower()
    if variant not in CUP_VARIANTS:
        known = ", ".join(sorted(CUP_VARIANTS))
        raise ValueError(f"未知杯子变体 {variant!r}；可选: {known}")
    return variant


def add_cup_cli_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cup",
        choices=tuple(CUP_VARIANTS.keys()),
        default=None,
        help=(
            f"杯子变体（默认 {DEFAULT_CUP_VARIANT}；"
            "也可用环境变量 PICKCUP_CUP_VARIANT=yellow|red）"
        ),
    )


def get_cup_variant_spec(cup_variant: str | None = None) -> dict[str, object]:
    return CUP_VARIANTS[resolve_cup_variant(cup_variant)]


@dataclass(frozen=True)
class SceneLayout:
    """一次 episode 的机器人/杯子布局。"""

    robot_position: tuple[float, float, float]
    cup_position: tuple[float, float, float]
    cup_yaw: float = 0.0
    robot_yaw: float = DEFAULT_ROBOT_YAW
    cup_variant: str = DEFAULT_CUP_VARIANT

    @classmethod
    def default(cls, cup_variant: str | None = None) -> SceneLayout:
        return cls(
            robot_position=tuple(DEFAULT_ROBOT_POSITION),
            cup_position=tuple(DEFAULT_CUP_POSITION),
            cup_yaw=0.0,
            robot_yaw=DEFAULT_ROBOT_YAW,
            cup_variant=resolve_cup_variant(cup_variant),
        )


def sample_random_scene(
    rng: np.random.Generator,
    *,
    cup_variant: str | None = None,
) -> SceneLayout:
    """采样随机桌面抓杯场景（机器人与杯子位置变化）。"""
    variant = resolve_cup_variant(cup_variant)
    rx = float(DEFAULT_ROBOT_POSITION[0] + rng.uniform(*ROBOT_X_RANGE))
    ry = float(DEFAULT_ROBOT_POSITION[1] + rng.uniform(*ROBOT_Y_RANGE))
    cx = float(DEFAULT_CUP_POSITION[0] + rng.uniform(*CUP_X_RANGE))
    cy = float(DEFAULT_CUP_POSITION[1] + rng.uniform(*CUP_Y_RANGE))
    yaw = float(rng.uniform(*CUP_YAW_RANGE))
    return SceneLayout(
        robot_position=(rx, ry, DEFAULT_ROBOT_POSITION[2]),
        cup_position=(cx, cy, DEFAULT_CUP_POSITION[2]),
        cup_yaw=yaw,
        robot_yaw=DEFAULT_ROBOT_YAW,
        cup_variant=variant,
    )


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def cup_yaw_to_quat(yaw: float) -> np.ndarray:
    return yaw_to_quat(yaw)


# 官方 G1 23DOF mode_10 关节顺序：腿 12 + 腰 1 + 左臂 5 + 右臂 5。
G1_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

TASK_GRIPPER_JOINTS = ["task_gripper_right", "task_gripper_left"]

JOINT_STEP = 0.05

# 官方 g1_23dof_mode_10 单臂 5DOF（见 unitree_ros g1_description/README.md）：
#   肩 pitch / roll / yaw + 肘 pitch + 腕 roll（无腕 pitch/yaw，那是 29DOF 机型）
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

# 朝 +Y 长边站位：在默认前伸姿态基础上，pitch 增大 + 肘展直，朝杯子靠近。
REACH_FORWARD_DELTAS = {
    "right_shoulder_pitch_joint": JOINT_STEP,
    "right_elbow_joint": -JOINT_STEP * 0.6,
    "right_shoulder_yaw_joint": JOINT_STEP * 0.4,
}
REACH_BACKWARD_DELTAS = {
    joint: -delta for joint, delta in REACH_FORWARD_DELTAS.items()
}
# 垂直抬/压：下压时肩 pitch 减小、肘弯曲增大。
REACH_UP_DELTAS = {
    "right_shoulder_pitch_joint": JOINT_STEP,
    "right_elbow_joint": -JOINT_STEP * 0.45,
}
REACH_DOWN_DELTAS = {
    "right_shoulder_pitch_joint": -JOINT_STEP,
    "right_elbow_joint": JOINT_STEP * 0.45,
}

KEY_CONTROL_JOINTS = [
    "waist_yaw_joint",
    *RIGHT_ARM_JOINTS,
]

# 训练/部署用 proprio（不含杯子位姿，与 action 维度对齐便于 ACT）。
PROPRIO_STATE_NAMES = KEY_CONTROL_JOINTS + TASK_GRIPPER_JOINTS
ACTION_NAMES = KEY_CONTROL_JOINTS + TASK_GRIPPER_JOINTS

# 完整状态（含杯子位姿），仅调试/分析用，不写入 LeRobot dataset。
FULL_STATE_NAMES = [
    "cup_x",
    "cup_y",
    "cup_z",
    "cup_vx",
    "cup_vy",
    "cup_vz",
    "cup_qw",
    "cup_qx",
    "cup_qy",
    "cup_qz",
    *PROPRIO_STATE_NAMES,
]
STATE_NAMES = FULL_STATE_NAMES

# 键盘/夹爪关节用运动学直接驱动；其余关节用 position actuator 保持姿态。
TELEOP_JOINTS = KEY_CONTROL_JOINTS + TASK_GRIPPER_JOINTS
HOLD_JOINTS = [joint for joint in G1_JOINTS if joint not in KEY_CONTROL_JOINTS]
ACTUATED_JOINTS = HOLD_JOINTS

GRIPPER_OPEN = {"task_gripper_right": -0.015, "task_gripper_left": 0.015}
GRIPPER_CLOSED = {"task_gripper_right": 0.010, "task_gripper_left": -0.010}

GRIPPER_GEOM_NAMES = frozenset({"task_gripper_right_geom", "task_gripper_left_geom"})
CUP_GEOM_NAMES = frozenset(
    {"cup_bottom", "cup_lower", "cup_mid1", "cup_mid2", "cup_rim"},
)
GRIPPER_ANCHOR_BODY = "right_wrist_roll_rubber_hand"
CONTACT_FRICTION = "1.0 0.005 0.0001"

# 抓取判定：要求手心朝向杯子、杯口在指间、手腕不过度扭转。
GRASP_MAX_WRIST_ROLL = 0.70
GRASP_MIN_APPROACH_DOT = 0.55
GRASP_MAX_DORSAL_DOT = -0.20
GRASP_MAX_LATERAL_DOT = 0.55
GRASP_MAX_FINGER_LATERAL_OFFSET = 0.042
GRASP_MAX_CUP_ABOVE_GRIPPER = 0.03
GRASP_MIN_CONTACT_FRONT_DOT = 0.45
GRASP_CONTACT_DIST_MAX = 0.002

TELEOP_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "right_wrist_roll_joint": (-GRASP_MAX_WRIST_ROLL, GRASP_MAX_WRIST_ROLL),
}

# 双臂对称待命：左右臂均为零位，右臂任务夹爪单独张开。
INITIAL_QPOS = {
    **{joint: 0.0 for joint in G1_JOINTS},
    **GRIPPER_OPEN,
}

KEY_BINDINGS = {
    glfw.KEY_W: ("waist_yaw_joint", JOINT_STEP),
    glfw.KEY_S: ("waist_yaw_joint", -JOINT_STEP),
    glfw.KEY_A: ("right_shoulder_roll_joint", JOINT_STEP),
    glfw.KEY_D: ("right_shoulder_roll_joint", -JOINT_STEP),
    glfw.KEY_Z: ("right_shoulder_yaw_joint", JOINT_STEP),
    glfw.KEY_X: ("right_shoulder_yaw_joint", -JOINT_STEP),
    glfw.KEY_C: ("right_elbow_joint", -JOINT_STEP),
    glfw.KEY_V: ("right_elbow_joint", JOINT_STEP),
    glfw.KEY_B: ("right_wrist_roll_joint", JOINT_STEP),
    glfw.KEY_N: ("right_wrist_roll_joint", -JOINT_STEP),
    glfw.KEY_T: ("right_shoulder_pitch_joint", JOINT_STEP),
    glfw.KEY_G: ("right_shoulder_pitch_joint", -JOINT_STEP),
}

# 兼容不同平台/输入法下传入 ASCII 字符码的情况。
ASCII_KEY_ALIASES = {
    ord("w"): glfw.KEY_W,
    ord("W"): glfw.KEY_W,
    ord("s"): glfw.KEY_S,
    ord("S"): glfw.KEY_S,
    ord("q"): glfw.KEY_Q,
    ord("Q"): glfw.KEY_Q,
    ord("e"): glfw.KEY_E,
    ord("E"): glfw.KEY_E,
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
    ord("h"): glfw.KEY_H,
    ord("H"): glfw.KEY_H,
    ord("r"): glfw.KEY_R,
    ord("R"): glfw.KEY_R,
    ord("p"): glfw.KEY_P,
    ord("P"): glfw.KEY_P,
}


def resolve_key(key: int | str) -> int:
    if isinstance(key, str):
        if key in {"BACKSPACE", "\x08"}:
            return glfw.KEY_BACKSPACE
        if key == " ":
            return glfw.KEY_SPACE
        if len(key) == 1:
            key = ord(key)
    if isinstance(key, int) and key in ASCII_KEY_ALIASES:
        key = ASCII_KEY_ALIASES[key]
    return key


def _fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _preprocess_g1_urdf() -> Path:
    """修正官方 URDF 的 meshdir/filename 双重 meshes 路径问题，不修改原文件。"""
    text = G1_URDF_PATH.read_text()
    text = text.replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        '<compiler discardvisual="false"/>',
    )

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix="g1_23dof_mode_10_preprocessed_",
        dir=G1_DESCRIPTION_DIR,
        delete=False,
    )
    with temp:
        temp.write(text)
    return Path(temp.name)


def _compile_g1_urdf_to_mjcf() -> ET.Element:
    """使用 MuJoCo 官方编译器把 G1 23DOF URDF 转为 MJCF XML 树。"""
    if (
        G1_COMPILED_MJCF_CACHE.exists()
        and G1_COMPILED_MJCF_CACHE.stat().st_mtime >= G1_URDF_PATH.stat().st_mtime
    ):
        return ET.parse(G1_COMPILED_MJCF_CACHE).getroot()

    preprocessed_urdf = _preprocess_g1_urdf()
    try:
        robot_model = mujoco.MjModel.from_xml_path(str(preprocessed_urdf))
        G1_COMPILED_MJCF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(G1_COMPILED_MJCF_CACHE), robot_model)
        root = ET.parse(G1_COMPILED_MJCF_CACHE).getroot()
    finally:
        preprocessed_urdf.unlink(missing_ok=True)

    root.set("model", "pick_cup_with_g1_23dof_mode_10")
    return root


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _set_joint_dynamics(root: ET.Element) -> None:
    for joint in root.findall(".//joint"):
        joint.set("damping", "2.0")
        joint.set("armature", "0.01")

        joint_name = joint.get("name", "")
        if joint_name in TELEOP_JOINTS:
            joint.set("damping", "1.0")
        elif joint_name in G1_JOINTS:
            joint.set("actuatorfrcrange", "-80 80")


def _get_joint_range(root: ET.Element, joint_name: str) -> tuple[float, float]:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"找不到关节: {joint_name}")

    range_text = joint.get("range")
    if not range_text:
        return -3.14, 3.14

    low, high = (float(value) for value in range_text.split()[:2])
    return low, high


def _add_scene_assets(root: ET.Element, cup_variant: str | None = None) -> None:
    compiler = _ensure_child(root, "compiler")
    compiler.set("angle", "radian")
    # XML 从字符串加载时没有文件基准目录，因此 meshdir 必须指向 g1_description。
    compiler.set("meshdir", str(G1_DESCRIPTION_DIR))

    asset = _ensure_child(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.85 0.88 0.92",
            "rgb2": "0.35 0.38 0.42",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "floor_texture",
            "type": "2d",
            "builtin": "flat",
            "rgb1": "0.62 0.62 0.60",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor_material",
            "texture": "floor_texture",
            "rgba": "0.62 0.62 0.60 1",
            "reflectance": "0",
        },
    )
    materials = {
        "table_white": "0.45 0.70 0.88 1",
        "leg_white": "0.88 0.88 0.90 1",
        "foot_black": "0.08 0.08 0.08 1",
        "task_gripper": "0.10 0.10 0.11 1",
    }
    cup_materials = get_cup_variant_spec(cup_variant)["materials"]
    assert isinstance(cup_materials, dict)
    materials.update(cup_materials)
    for name, rgba in materials.items():
        attrs = {"name": name, "rgba": str(rgba)}
        if name.startswith("cup_"):
            # 金属杯略提高高光，便于视觉辨认光滑曲面。
            attrs["specular"] = "0.5"
            attrs["shininess"] = "0.55"
            attrs["reflectance"] = "0.08"
        ET.SubElement(asset, "material", attrs)

    # 视觉截锥 mesh（绝对路径，不受 G1 meshdir 影响）。
    _register_cup_visual_meshes(asset, cup_variant)


def _register_cup_visual_meshes(asset: ET.Element, cup_variant: str | None = None) -> None:
    variant = resolve_cup_variant(cup_variant)
    segments = get_cup_variant_spec(variant)["segments"]
    assert isinstance(segments, tuple)
    profile = profile_from_segments(segments)
    mesh_paths = ensure_cup_visual_meshes(variant, profile)
    for part, path in mesh_paths.items():
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": f"cup_visual_{part}_{variant}",
                "file": str(path),
                # 口沿等薄片体积近 0，用 shell 惯性避免编译报错（视觉 geom density=0）。
                "inertia": "shell",
            },
        )


def _add_world_options(root: ET.Element) -> None:
    option = _ensure_child(root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "RK4")

    statistic = _ensure_child(root, "statistic")
    statistic.set("center", "0 -0.2 0.8")
    statistic.set("extent", "1.8")

    visual = _ensure_child(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.35 0.35 0.35",
            "diffuse": "0.7 0.7 0.7",
            "specular": "0.15 0.15 0.15",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.18 0.22 1"})
    ET.SubElement(
        visual,
        "global",
        {"azimuth": "-135", "elevation": "-20", "offwidth": "960", "offheight": "720"},
    )
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "50"})
    # 将关节/接触/约束等装饰尺寸置零，避免 passive viewer 重置后仍出现拉伸圆柱。
    ET.SubElement(
        visual,
        "scale",
        {
            "forcewidth": "0",
            "contactwidth": "0",
            "contactheight": "0",
            "connect": "0",
            "com": "0",
            "camera": "0",
            "light": "0",
            "selectpoint": "0",
            "jointlength": "0",
            "jointwidth": "0",
            "actuatorlength": "0",
            "actuatorwidth": "0",
            "framelength": "0",
            "framewidth": "0",
            "constraint": "0",
            "slidercrank": "0",
        },
    )


def _wrap_robot_in_fixed_pelvis(
    root: ET.Element,
    robot_position: Sequence[float],
    robot_yaw: float,
) -> ET.Element:
    worldbody = _ensure_child(root, "worldbody")
    robot_nodes = list(worldbody)

    for child in robot_nodes:
        worldbody.remove(child)

    pelvis = ET.Element(
        "body",
        {
            "name": "pelvis",
            "pos": _fmt(robot_position),
            "quat": _fmt(yaw_to_quat(robot_yaw)),
        },
    )
    for child in robot_nodes:
        pelvis.append(child)
    worldbody.append(pelvis)
    return pelvis


def _contact_attrs() -> dict[str, str]:
    return {
        "contype": "1",
        "conaffinity": "1",
        "friction": CONTACT_FRICTION,
        "condim": "4",
        "solref": "0.02 1",
        "solimp": "0.9 0.95 0.001",
    }


def _add_task_gripper(pelvis: ET.Element) -> None:
    wrist = pelvis.find(".//body[@name='right_wrist_roll_rubber_hand']")
    if wrist is None:
        raise ValueError("找不到 right_wrist_roll_rubber_hand，无法添加任务夹爪")

    right_body = ET.SubElement(
        wrist,
        "body",
        {"name": "task_gripper_right_body", "pos": "0.175 -0.045 0.005"},
    )
    ET.SubElement(
        right_body,
        "joint",
        {
            "name": "task_gripper_right",
            "type": "slide",
            "axis": "0 1 0",
            "range": "-0.018 0.016",
            "damping": "1.0",
            "armature": "0.001",
        },
    )
    ET.SubElement(
        right_body,
        "geom",
        {
            "name": "task_gripper_right_geom",
            "type": "box",
            "size": "0.035 0.008 0.014",
            "pos": "0.025 0 0",
            "material": "task_gripper",
            **_contact_attrs(),
        },
    )

    left_body = ET.SubElement(
        wrist,
        "body",
        {"name": "task_gripper_left_body", "pos": "0.175 0.045 0.005"},
    )
    ET.SubElement(
        left_body,
        "joint",
        {
            "name": "task_gripper_left",
            "type": "slide",
            "axis": "0 1 0",
            "range": "-0.016 0.018",
            "damping": "1.0",
            "armature": "0.001",
        },
    )
    ET.SubElement(
        left_body,
        "geom",
        {
            "name": "task_gripper_left_geom",
            "type": "box",
            "size": "0.035 0.008 0.014",
            "pos": "0.025 0 0",
            "material": "task_gripper",
            **_contact_attrs(),
        },
    )


def _add_task_world(
    root: ET.Element,
    cup_position: Sequence[float],
    cup_variant: str | None = None,
) -> None:
    worldbody = _ensure_child(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": "-1.0 -1.4 3.0",
            "dir": "0.4 0.5 -1",
            "directional": "true",
            "diffuse": "0.85 0.85 0.8",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill_light",
            "pos": "1.5 1.2 2.2",
            "dir": "-0.5 -0.3 -1",
            "directional": "true",
            "diffuse": "0.25 0.25 0.3",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "4 4 0.1",
            "material": "floor_material",
            "contype": "1",
            "conaffinity": "1",
            "friction": CONTACT_FRICTION,
            "condim": "4",
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.001",
        },
    )

    table_top = ET.SubElement(worldbody, "body", {"name": "table_top", "pos": f"0 0 {TABLE_HEIGHT}"})
    ET.SubElement(
        table_top,
        "geom",
        {
            "name": "table_surface",
            "type": "box",
            "size": _fmt((TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_THICKNESS / 2)),
            "material": "table_white",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    for name, x, y in [
        ("leg1", -0.68, -0.24),
        ("leg2", 0.68, -0.24),
        ("leg3", -0.68, 0.24),
        ("leg4", 0.68, 0.24),
    ]:
        leg = ET.SubElement(table_top, "body", {"name": name, "pos": f"{x} {y} 0"})
        ET.SubElement(
            leg,
            "geom",
            {
                "type": "cylinder",
                "size": "0.02 0.36875",
                "material": "leg_white",
                "pos": "0 0 -0.36875",
            },
        )
        ET.SubElement(
            leg,
            "geom",
            {
                "type": "cylinder",
                "size": "0.025 0.015",
                "material": "foot_black",
                "pos": "0 0 -0.7375",
            },
        )

    variant = resolve_cup_variant(cup_variant)
    segments = get_cup_variant_spec(variant)["segments"]
    assert isinstance(segments, tuple)
    cup = ET.SubElement(worldbody, "body", {"name": "cup", "pos": _fmt(cup_position)})
    ET.SubElement(cup, "joint", {"name": "cup_free", "type": "free", "damping": "0.15", "armature": "0.001"})

    # 视觉 mesh 局部 z∈[0,H]，对齐到碰撞分段底面。
    visual_z0 = float(segments[0][5]) - float(segments[0][2])
    visual_parts = (
        ("outer", "cup_outer"),
        ("inner", "cup_inner"),
        ("bottom", "cup_outer"),
        ("rim", "cup_inner"),
    )
    for part, material in visual_parts:
        ET.SubElement(
            cup,
            "geom",
            {
                "name": f"cup_visual_{part}",
                "type": "mesh",
                "mesh": f"cup_visual_{part}_{variant}",
                "material": material,
                "pos": f"0 0 {visual_z0:.6g}",
                "contype": "0",
                "conaffinity": "0",
                "group": "0",
                "density": "0",
            },
        )

    # 碰撞：隐藏圆柱分段（保留原 geom 名，抓取逻辑不变）。
    for name, radius, half_height, _material, mass, z in segments:
        ET.SubElement(
            cup,
            "geom",
            {
                "name": name,
                "type": "cylinder",
                "size": _fmt((radius, half_height)),
                "mass": f"{mass:.6g}",
                "pos": f"0 0 {z:.6g}",
                "group": "3",
                "rgba": "0 0 0 0",
                **_contact_attrs(),
            },
        )

# 场景物体保留碰撞；机器人仅任务夹爪保留碰撞，避免 teleop 时出现拉伸 hull/接触力可视化。
SCENE_COLLISION_GEOM_NAMES = {
    "floor",
    "table_surface",
    "cup_bottom",
    "cup_lower",
    "cup_mid1",
    "cup_mid2",
    "cup_rim",
    "task_gripper_right_geom",
    "task_gripper_left_geom",
}

# 参与 teleop 碰撞检测的 link：保留隐藏 collision mesh（group=3，不参与渲染）。
TELEOP_COLLISION_BODIES = frozenset(
    {
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_rubber_hand",
        "task_gripper_right_body",
        "task_gripper_left_body",
    }
)

OBSTACLE_BODY_NAMES = frozenset(
    {
        "table_top",
        "leg1",
        "leg2",
        "leg3",
        "leg4",
    }
)

COLLISION_SUBSTEPS = 16
MAX_PENETRATION = 0.003


def _sanitize_robot_geoms(root: ET.Element) -> None:
    """去掉多余碰撞 proxy；teleop 相关 link 保留隐藏 collision mesh 供碰撞检测。"""
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is None:
        return

    for body in (pelvis, *pelvis.iter("body")):
        body_name = body.get("name", "")
        keep_collision = body_name in TELEOP_COLLISION_BODIES
        visual_meshes = {
            geom.get("mesh")
            for geom in body.findall("geom")
            if geom.get("mesh") and geom.get("contype") == "0"
        }
        to_remove: list[ET.Element] = []
        for geom in body.findall("geom"):
            name = geom.get("name", "")
            if name in SCENE_COLLISION_GEOM_NAMES:
                continue

            mesh = geom.get("mesh")
            if mesh and geom.get("contype") != "0" and mesh in visual_meshes:
                if keep_collision:
                    geom.set("group", "4")
                    geom.set("contype", "1")
                    geom.set("conaffinity", "1")
                    geom.set("density", "0")
                    geom.set("rgba", "0 0 0 0")
                else:
                    to_remove.append(geom)
                continue

            geom_type = geom.get("type", "mesh" if mesh else "sphere")
            if geom_type in ("cylinder", "sphere", "capsule"):
                to_remove.append(geom)

        for geom in to_remove:
            body.remove(geom)


def _configure_viewer(viewer: mujoco.viewer.Handle) -> None:
    """关闭调试可视化。passive viewer 的 sync() 每帧会重置 opt，因此需在主循环内反复调用。"""
    disable_flags = (
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT,
        mujoco.mjtVisFlag.mjVIS_JOINT,
        mujoco.mjtVisFlag.mjVIS_CONVEXHULL,
        mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        mujoco.mjtVisFlag.mjVIS_AUTOCONNECT,
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_SCLINERTIA,
        mujoco.mjtVisFlag.mjVIS_COM,
        mujoco.mjtVisFlag.mjVIS_PERTFORCE,
        mujoco.mjtVisFlag.mjVIS_PERTOBJ,
        mujoco.mjtVisFlag.mjVIS_BODYBVH,
        mujoco.mjtVisFlag.mjVIS_MESHBVH,
        mujoco.mjtVisFlag.mjVIS_CAMERA,
        mujoco.mjtVisFlag.mjVIS_LIGHT,
        mujoco.mjtVisFlag.mjVIS_SELECT,
        mujoco.mjtVisFlag.mjVIS_ACTUATOR,
        mujoco.mjtVisFlag.mjVIS_TENDON,
        mujoco.mjtVisFlag.mjVIS_RANGEFINDER,
    )
    for flag in disable_flags:
        viewer.opt.flags[flag] = 0

    # 桌子/桌腿是无 joint 的 static body，关闭此 flag 会导致它们不渲染。
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
    viewer.opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
    viewer.perturb.active = 0


def _add_head_camera(
    root: ET.Element,
    robot_position: Sequence[float],
    cup_position: Sequence[float],
    robot_yaw: float,
) -> None:
    """按 G1-23 官方 D435 安装外参挂载真机标定的 RGB 相机。"""
    torso = root.find(".//body[@name='torso_link']")
    if torso is None:
        raise ValueError("找不到 torso_link，无法添加 head_camera")

    d435_body = ET.SubElement(
        torso,
        "body",
        {"name": "d435_link", "pos": _fmt(D435_TORSO_POSITION)},
    )
    ET.SubElement(
        d435_body,
        "camera",
        {
            "name": HEAD_CAMERA_NAME,
            "pos": "0 0 0",
            "quat": _fmt(D435_MUJOCO_QUATERNION),
            "resolution": " ".join(map(str, D435_COLOR_RESOLUTION)),
            "sensorsize": _fmt(D435_COLOR_SENSOR_SIZE),
            "focalpixel": _fmt(D435_COLOR_FOCAL_PIXEL),
            "principalpixel": _fmt(D435_COLOR_PRINCIPAL_PIXEL),
            "mode": "fixed",
        },
    )


def _add_position_actuators(root: ET.Element) -> None:
    old_actuator = root.find("actuator")
    if old_actuator is not None:
        root.remove(old_actuator)

    actuator = ET.SubElement(root, "actuator")
    for joint_name in HOLD_JOINTS:
        low, high = _get_joint_range(root, joint_name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_pos",
                "joint": joint_name,
                "kp": "120",
                "ctrlrange": _fmt((low, high)),
                "forcelimited": "true",
                "forcerange": "-80 80",
            },
        )


def build_xml(scene: SceneLayout | None = None) -> str:
    """从官方 G1 23DOF URDF 生成完整 pick-cup MJCF。"""
    layout = scene or SceneLayout.default()
    variant = resolve_cup_variant(layout.cup_variant)
    root = _compile_g1_urdf_to_mjcf()
    _set_joint_dynamics(root)
    _add_scene_assets(root, variant)
    _add_world_options(root)
    pelvis = _wrap_robot_in_fixed_pelvis(root, layout.robot_position, layout.robot_yaw)
    _add_task_gripper(pelvis)
    _add_task_world(root, layout.cup_position, variant)
    _sanitize_robot_geoms(root)
    _add_head_camera(root, layout.robot_position, layout.cup_position, layout.robot_yaw)
    _add_position_actuators(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def make_model_data(scene: SceneLayout | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(build_xml(scene))
    return model, mujoco.MjData(model)


def get_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo 对象不存在: {name}")
    return obj_id


def collect_cup_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    cup_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
    qpos_addr = model.jnt_qposadr[cup_joint_id]
    qvel_addr = model.jnt_dofadr[cup_joint_id]

    pos = data.qpos[qpos_addr : qpos_addr + 3]
    quat = data.qpos[qpos_addr + 3 : qpos_addr + 7]
    vel = data.qvel[qvel_addr : qvel_addr + 6]

    return {
        "cup_x": float(pos[0]),
        "cup_y": float(pos[1]),
        "cup_z": float(pos[2]),
        "cup_vx": float(vel[0]),
        "cup_vy": float(vel[1]),
        "cup_vz": float(vel[2]),
        "cup_qw": float(quat[0]),
        "cup_qx": float(quat[1]),
        "cup_qy": float(quat[2]),
        "cup_qz": float(quat[3]),
    }


def collect_proprio_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """8 维 proprio：腰 + 右臂 + 夹爪（不含杯子位姿，供 ACT 训练/部署）。"""
    values = {
        joint_name: get_joint_qpos(model, data, joint_name)
        for joint_name in PROPRIO_STATE_NAMES
    }
    return np.array([values[name] for name in PROPRIO_STATE_NAMES], dtype=np.float32)


def collect_observation_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    values = collect_cup_state(model, data)
    for joint_name in PROPRIO_STATE_NAMES:
        values[joint_name] = get_joint_qpos(model, data, joint_name)
    return np.array([values[name] for name in FULL_STATE_NAMES], dtype=np.float32)


def collect_action_vector(robot_ctrl: dict[str, float]) -> np.ndarray:
    return np.array([robot_ctrl[name] for name in ACTION_NAMES], dtype=np.float32)


def render_head_camera(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
) -> np.ndarray:
    renderer.update_scene(data, camera=HEAD_CAMERA_NAME)
    return renderer.render()


def _robot_view_overlay_lines() -> tuple[str, ...]:
    return (
        f"{D435_MODEL}  SN:{D435_SERIAL}",
        f"FW:{D435_FIRMWARE}  (rec {D435_FIRMWARE_RECOMMENDED})",
        f"{D435_COLOR_RESOLUTION[0]}x{D435_COLOR_RESOLUTION[1]}"
        f"  FOV {D435_COLOR_HFOV_DEG:.2f} x {D435_COLOR_VFOV_DEG:.2f} deg",
        f"fx={D435_COLOR_FX:.4f}  fy={D435_COLOR_FY:.4f}",
        f"cx={D435_COLOR_CX:.4f}  cy={D435_COLOR_CY:.4f}",
    )


def set_viewer_robot_camera(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    enabled: bool,
) -> None:
    """将 MuJoCo Viewer 切换到机器人 head_camera，或回到自由相机。"""
    if enabled:
        cam_id = get_id(model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def init_head_camera_preview(*, enable_by_default: bool = True) -> bool:
    """在普通 Python 子进程创建预览，避免 mjpython 与 Qt/Cocoa GUI 冲突。"""
    global _head_camera_preview, _head_camera_preview_enabled
    default = "1" if enable_by_default else "0"
    if os.environ.get("PICKCUP_HEAD_CAMERA_PREVIEW", default) not in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }:
        _head_camera_preview_enabled = False
        print("机器人视角预览已禁用；设置 PICKCUP_HEAD_CAMERA_PREVIEW=1 可开启")
        return False

    if _head_camera_preview is not None and _head_camera_preview.is_alive:
        _head_camera_preview_enabled = True
        return True

    preview = HeadCameraPreviewProcess(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        HEAD_CAMERA_WINDOW,
        overlay_lines=_robot_view_overlay_lines(),
    )
    if not preview.start():
        preview.close()
        _head_camera_preview = None
        _head_camera_preview_enabled = False
        print("机器人视角预览已禁用（独立预览进程启动失败）")
        return False

    _head_camera_preview = preview
    _head_camera_preview_enabled = True
    print(
        f"机器人视角预览已开启（独立进程）: {HEAD_CAMERA_WINDOW} "
        f"({CAMERA_WIDTH}x{CAMERA_HEIGHT}, "
        f"FOV {D435_COLOR_HFOV_DEG:.2f}°×{D435_COLOR_VFOV_DEG:.2f}°)"
    )
    return True


_head_camera_preview: HeadCameraPreviewProcess | None = None
_head_camera_preview_enabled = False


def show_head_camera_preview(rgb: np.ndarray) -> None:
    global _head_camera_preview, _head_camera_preview_enabled
    if not _head_camera_preview_enabled:
        return

    if _head_camera_preview is None or not _head_camera_preview.publish(rgb):
        _head_camera_preview_enabled = False
        print("机器人视角预览进程已关闭；3D Viewer 与录制继续运行")
        if _head_camera_preview is not None:
            _head_camera_preview.close()
            _head_camera_preview = None


def close_head_camera_preview() -> None:
    global _head_camera_preview, _head_camera_preview_enabled
    _head_camera_preview_enabled = False
    if _head_camera_preview is not None:
        _head_camera_preview.close()
        _head_camera_preview = None


def get_joint_range(model: mujoco.MjModel, joint_name: str) -> tuple[float, float]:
    joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if not model.jnt_limited[joint_id]:
        return -np.inf, np.inf
    return float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1])


def get_teleop_joint_range(model: mujoco.MjModel, joint_name: str) -> tuple[float, float]:
    """Teleop 关节范围；手腕 roll 额外收紧，避免手背/扭曲姿态抓取。"""
    low, high = get_joint_range(model, joint_name)
    if joint_name in TELEOP_JOINT_LIMITS:
        limit_low, limit_high = TELEOP_JOINT_LIMITS[joint_name]
        low = max(low, limit_low)
        high = min(high, limit_high)
    return low, high


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr] = value


def adjust_joint_target(
    model: mujoco.MjModel,
    robot_ctrl: dict[str, float],
    joint_name: str,
    value: float,
) -> float:
    low, high = get_teleop_joint_range(model, joint_name)
    clipped = float(np.clip(value, low, high))
    robot_ctrl[joint_name] = clipped
    return clipped


def build_collision_geom_sets(model: mujoco.MjModel) -> tuple[frozenset[int], frozenset[int]]:
    body_ids_for = lambda names: {
        get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in names
    }
    teleop_body_ids = body_ids_for(TELEOP_COLLISION_BODIES)
    obstacle_body_ids = body_ids_for(OBSTACLE_BODY_NAMES)

    teleop_geoms: set[int] = set()
    obstacle_geoms: set[int] = set()
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] == 0:
            continue
        body_id = model.geom_bodyid[geom_id]
        if body_id in teleop_body_ids:
            teleop_geoms.add(geom_id)
        if body_id in obstacle_body_ids:
            obstacle_geoms.add(geom_id)
    return frozenset(teleop_geoms), frozenset(obstacle_geoms)


def build_named_geom_set(model: mujoco.MjModel, geom_names: frozenset[str]) -> frozenset[int]:
    return frozenset(
        get_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in geom_names
    )


def is_gripper_closed(robot_ctrl: dict[str, float]) -> bool:
    return (
        robot_ctrl["task_gripper_right"] >= GRIPPER_CLOSED["task_gripper_right"] - 0.003
        and robot_ctrl["task_gripper_left"] <= GRIPPER_CLOSED["task_gripper_left"] + 0.003
    )


def is_gripper_open(robot_ctrl: dict[str, float]) -> bool:
    return (
        robot_ctrl["task_gripper_right"] <= GRIPPER_OPEN["task_gripper_right"] + 0.003
        and robot_ctrl["task_gripper_left"] >= GRIPPER_OPEN["task_gripper_left"] - 0.003
    )


def geoms_in_contact(data: mujoco.MjData, geoms_a: frozenset[int], geoms_b: frozenset[int]) -> bool:
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        if (
            contact.geom1 in geoms_a and contact.geom2 in geoms_b
        ) or (
            contact.geom2 in geoms_a and contact.geom1 in geoms_b
        ):
            if contact.dist <= 0.002:
                return True
    return False


def get_grip_anchor_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    body_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_ANCHOR_BODY)
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def get_cup_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    cup_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
    qpos_addr = model.jnt_qposadr[cup_joint_id]
    pos = data.qpos[qpos_addr : qpos_addr + 3].copy()
    quat = data.qpos[qpos_addr + 3 : qpos_addr + 7].copy()
    return pos, quat


def set_cup_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    quat: np.ndarray,
) -> None:
    cup_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
    qpos_addr = model.jnt_qposadr[cup_joint_id]
    qvel_addr = model.jnt_dofadr[cup_joint_id]
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    if qvel_addr >= 0:
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def _cup_over_table(cup_pos: np.ndarray) -> bool:
    return (
        abs(float(cup_pos[0])) <= TABLE_LENGTH / 2 + 0.04
        and abs(float(cup_pos[1])) <= TABLE_WIDTH / 2 + 0.04
    )


def sanitize_cup_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    grasp_active: bool = False,
) -> None:
    """防止杯子穿透地面或掉到桌面以下（未吸附时）。"""
    if grasp_active:
        return

    cup_pos, cup_quat = get_cup_pose(model, data)
    cup_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
    qvel_addr = model.jnt_dofadr[cup_joint_id]
    if qvel_addr < 0:
        return

    if _cup_over_table(cup_pos):
        min_z = CUP_MIN_WORLD_Z
        target_z = TABLE_SURFACE_Z + 0.0025
    else:
        min_z = FLOOR_Z + 0.03
        target_z = min_z

    if cup_pos[2] >= min_z:
        return

    cup_pos = cup_pos.copy()
    cup_pos[2] = target_z
    set_cup_pose(model, data, cup_pos, cup_quat)
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


@dataclass
class GraspState:
    active: bool = False
    local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_mat: np.ndarray = field(default_factory=lambda: np.eye(3))

    def reset(self) -> None:
        self.active = False
        self.local_pos[:] = 0.0
        self.local_mat[:] = np.eye(3)


def get_gripper_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    right_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, "task_gripper_right_body")
    left_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, "task_gripper_left_body")
    return 0.5 * (data.xpos[right_id] + data.xpos[left_id])


def cup_near_gripper(model: mujoco.MjModel, data: mujoco.MjData, max_dist: float = 0.09) -> bool:
    cup_pos, _ = get_cup_pose(model, data)
    gripper_center = get_gripper_center(model, data)
    if cup_pos[2] < TABLE_SURFACE_Z - 0.01:
        return False
    return float(np.linalg.norm(cup_pos - gripper_center)) < max_dist


def _gripper_cup_contacts(
    data: mujoco.MjData,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> list[tuple[object, bool]]:
    contacts: list[tuple[object, bool]] = []
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        if contact.geom1 in gripper_geoms and contact.geom2 in cup_geoms:
            contacts.append((contact, False))
        elif contact.geom2 in gripper_geoms and contact.geom1 in cup_geoms:
            contacts.append((contact, True))
    return contacts


def _grasp_alignment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float]:
    anchor_pos, anchor_mat = get_grip_anchor_pose(model, data)
    gripper_center = get_gripper_center(model, data)
    cup_pos, _ = get_cup_pose(model, data)
    to_cup = cup_pos - gripper_center
    to_cup_dist = float(np.linalg.norm(to_cup))
    to_cup_unit = to_cup / max(to_cup_dist, 1e-9)
    approach_axis = anchor_mat[:, 0]
    lateral_axis = anchor_mat[:, 1]
    dorsal_axis = anchor_mat[:, 2]
    return (
        anchor_mat,
        gripper_center,
        cup_pos,
        to_cup_unit,
        float(approach_axis @ to_cup_unit),
        float(dorsal_axis @ to_cup_unit),
        float(abs(lateral_axis @ to_cup_unit)),
        float(abs(np.dot(cup_pos - gripper_center, lateral_axis))),
        float(cup_pos[2] - gripper_center[2]),
    )


def has_valid_grasp_contact(
    data: mujoco.MjData,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
    anchor_mat: np.ndarray,
    gripper_center: np.ndarray,
) -> bool:
    """夹爪指面与杯壁的有效接触：接触法向应来自手心前方，接触点在指间。"""
    approach_axis = anchor_mat[:, 0]
    lateral_axis = anchor_mat[:, 1]
    for contact, _flip_normal in _gripper_cup_contacts(data, gripper_geoms, cup_geoms):
        if contact.dist > GRASP_CONTACT_DIST_MAX:
            continue
        if contact.geom1 in gripper_geoms:
            grip_to_cup = contact.frame[:3].copy()
        else:
            grip_to_cup = -contact.frame[:3].copy()
        front_dot = float(grip_to_cup @ approach_axis)
        lateral_offset = float(abs(np.dot(contact.pos - gripper_center, lateral_axis)))
        if (
            front_dot >= GRASP_MIN_CONTACT_FRONT_DOT
            and lateral_offset <= GRASP_MAX_FINGER_LATERAL_OFFSET
        ):
            return True
    return False


def _grasp_alignment_ok(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> bool:
    """手心朝向杯子、指间有效接触、手腕未过度扭转（不含夹爪开/合检查）。"""
    wrist_roll = get_joint_qpos(model, data, "right_wrist_roll_joint")
    if abs(wrist_roll) > GRASP_MAX_WRIST_ROLL + 1e-6:
        return False

    (
        anchor_mat,
        gripper_center,
        _cup_pos,
        _to_cup_unit,
        approach_dot,
        dorsal_dot,
        lateral_dot,
        finger_lateral_offset,
        cup_above_gripper,
    ) = _grasp_alignment(model, data)

    if approach_dot < GRASP_MIN_APPROACH_DOT:
        return False
    if dorsal_dot > GRASP_MAX_DORSAL_DOT:
        return False
    if lateral_dot > GRASP_MAX_LATERAL_DOT:
        return False
    if finger_lateral_offset > GRASP_MAX_FINGER_LATERAL_OFFSET:
        return False
    if cup_above_gripper > GRASP_MAX_CUP_ABOVE_GRIPPER:
        return False
    return has_valid_grasp_contact(
        data,
        gripper_geoms,
        cup_geoms,
        anchor_mat,
        gripper_center,
    )


def can_initiate_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> bool:
    """是否满足“手心朝向杯子、指间夹持、手腕未扭曲”的抓取条件。"""
    if not is_gripper_closed(robot_ctrl):
        return False
    return _grasp_alignment_ok(model, data, gripper_geoms, cup_geoms)


def can_maintain_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> bool:
    """已抓住后：与 initiate 相同标准，防止抬起/保持阶段手背或扭曲姿态仍算成功。"""
    if not is_gripper_closed(robot_ctrl):
        return False
    return _grasp_alignment_ok(model, data, gripper_geoms, cup_geoms)


def update_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    grasp: GraspState,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> None:
    """teleop 模式下：手心朝向杯子的有效指间夹持才吸附，张开或姿态失效时释放。"""
    anchor_pos, anchor_mat = get_grip_anchor_pose(model, data)

    if grasp.active:
        if is_gripper_open(robot_ctrl) or not can_maintain_grasp(
            model,
            data,
            robot_ctrl,
            gripper_geoms,
            cup_geoms,
        ):
            grasp.reset()
            print(">>> 已释放杯子")
            return

        cup_mat = anchor_mat @ grasp.local_mat
        cup_pos = anchor_pos + anchor_mat @ grasp.local_pos
        cup_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cup_quat, cup_mat.reshape(-1))
        set_cup_pose(model, data, cup_pos, cup_quat)
        return

    if can_initiate_grasp(model, data, robot_ctrl, gripper_geoms, cup_geoms):
        cup_pos, cup_quat = get_cup_pose(model, data)
        cup_mat = np.zeros(9)
        mujoco.mju_quat2Mat(cup_mat, cup_quat)
        grasp.local_pos = anchor_mat.T @ (cup_pos - anchor_pos)
        grasp.local_mat = anchor_mat.T @ cup_mat.reshape(3, 3)
        grasp.active = True
        print(">>> 已抓住杯子")


def measure_teleop_obstacle_penetration(
    data: mujoco.MjData,
    teleop_geoms: frozenset[int],
    obstacle_geoms: frozenset[int],
) -> float:
    max_penetration = 0.0
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        if not (
            (contact.geom1 in teleop_geoms and contact.geom2 in obstacle_geoms)
            or (contact.geom2 in teleop_geoms and contact.geom1 in obstacle_geoms)
        ):
            continue
        if contact.dist < 0.0:
            max_penetration = max(max_penetration, -contact.dist)
    return max_penetration


def penetration_exceeds_limit(penetration: float) -> bool:
    return penetration > MAX_PENETRATION + 1e-4


def try_update_teleop_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    updates: dict[str, float],
    teleop_geoms: frozenset[int],
    obstacle_geoms: frozenset[int],
) -> bool:
    """按子步尝试更新 teleop 目标；若加深与桌子/杯子的穿透则回退并返回 False。"""
    if not updates:
        return True

    apply_kinematic_teleop(model, data, robot_ctrl)
    mujoco.mj_collision(model, data)

    snapshot_qpos = data.qpos.copy()
    last_good_qpos = snapshot_qpos.copy()
    last_good_ctrl = {name: robot_ctrl[name] for name in TELEOP_JOINTS}
    moved = False
    targets = {
        joint_name: float(
            np.clip(value, *get_teleop_joint_range(model, joint_name)),
        )
        for joint_name, value in updates.items()
    }
    starts = {joint_name: robot_ctrl[joint_name] for joint_name in targets}

    for step in range(1, COLLISION_SUBSTEPS + 1):
        alpha = step / COLLISION_SUBSTEPS
        for joint_name, target in targets.items():
            start = starts[joint_name]
            robot_ctrl[joint_name] = start + (target - start) * alpha
        apply_kinematic_teleop(model, data, robot_ctrl)
        mujoco.mj_collision(model, data)
        penetration_after = measure_teleop_obstacle_penetration(
            data,
            teleop_geoms,
            obstacle_geoms,
        )
        if penetration_exceeds_limit(penetration_after):
            data.qpos[:] = last_good_qpos
            for joint_name, value in last_good_ctrl.items():
                robot_ctrl[joint_name] = value
            apply_kinematic_teleop(model, data, robot_ctrl)
            return moved

        moved = True
        last_good_qpos = data.qpos.copy()
        last_good_ctrl = {name: robot_ctrl[name] for name in TELEOP_JOINTS}

    return True


def apply_kinematic_teleop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
) -> None:
    """键盘控制关节直接写入 qpos，保证按键后立刻可见。"""
    for joint_name in TELEOP_JOINTS:
        low, high = get_teleop_joint_range(model, joint_name)
        value = float(np.clip(robot_ctrl[joint_name], low, high))
        robot_ctrl[joint_name] = value
        joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = model.jnt_qposadr[joint_id]
        dof_addr = model.jnt_dofadr[joint_id]
        data.qpos[qpos_addr] = value
        if dof_addr >= 0:
            data.qvel[dof_addr] = 0.0
    mujoco.mj_forward(model, data)


def get_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    return float(data.qpos[qpos_addr])


def get_actuator_indices(model: mujoco.MjModel) -> dict[str, int]:
    return {
        joint_name: get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_pos")
        for joint_name in ACTUATED_JOINTS
    }


def reset_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    actuator_indices: dict[str, int],
    *,
    scene: SceneLayout | None = None,
) -> None:
    mujoco.mj_resetData(model, data)
    for joint_name, value in INITIAL_QPOS.items():
        robot_ctrl[joint_name] = value
        set_joint_qpos(model, data, joint_name, value)
    layout = scene or SceneLayout.default()
    set_cup_pose(
        model,
        data,
        np.array(layout.cup_position, dtype=np.float64),
        cup_yaw_to_quat(layout.cup_yaw),
    )
    for joint_name in HOLD_JOINTS:
        data.ctrl[actuator_indices[joint_name]] = robot_ctrl[joint_name]
    apply_kinematic_teleop(model, data, robot_ctrl)
    mujoco.mj_forward(model, data)


def apply_hold_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    actuator_indices: dict[str, int],
) -> None:
    for joint_name in HOLD_JOINTS:
        low, high = get_joint_range(model, joint_name)
        clipped = float(np.clip(robot_ctrl[joint_name], low, high))
        robot_ctrl[joint_name] = clipped
        data.ctrl[actuator_indices[joint_name]] = clipped


def maybe_record_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    renderer: mujoco.Renderer,
    recorder: PickcupLeRobotRecorder,
    next_record_time: float,
) -> tuple[float, np.ndarray | None]:
    """按仿真时间 30Hz 采样；返回 (next_record_time, 最新 RGB 供预览)。"""
    last_rgb: np.ndarray | None = None
    while data.time + 1e-9 >= next_record_time:
        rgb = render_head_camera(renderer, data)
        last_rgb = rgb
        recorder.add_frame(
            observation_state=collect_proprio_state(model, data),
            action=collect_action_vector(robot_ctrl),
            rgb=rgb,
            task=DEFAULT_TASK,
        )
        next_record_time += RECORD_PERIOD
    return next_record_time, last_rgb


def print_help(*, cup_variant: str | None = None) -> None:
    variant = resolve_cup_variant(cup_variant)
    variant_label = get_cup_variant_spec(variant)["label"]
    print("=" * 60)
    print("MuJoCo G1 23DOF 杯子抓取仿真场景")
    print("=" * 60)
    print("\n场景配置:")
    print(f"  - 机器人模型: {G1_URDF_PATH}")
    print("  - 机器人: Unitree G1 23DOF mode_10，pelvis 固定")
    print(f"  - 桌子: {TABLE_LENGTH * 100:.0f}cm x {TABLE_WIDTH * 100:.0f}cm x {TABLE_HEIGHT * 100:.0f}cm")
    print(f"  - 杯子变体: {variant}（{variant_label}）")
    print(f"  - 杯子位置: ({CUP_POSITION[0]:.2f}, {CUP_POSITION[1]:.2f}, {CUP_POSITION[2]:.3f})")
    print(f"  - 机器人位置: ({ROBOT_POSITION[0]:.2f}, {ROBOT_POSITION[1]:.2f}, {ROBOT_POSITION[2]:.3f})")
    print("\n头部相机 (真机 D435i):")
    print(f"  - 设备: {D435_MODEL}  SN={D435_SERIAL}")
    print(f"  - 固件: {D435_FIRMWARE}（官方建议 {D435_FIRMWARE_RECOMMENDED}）")
    print(
        f"  - 彩色: {D435_COLOR_RESOLUTION[0]}x{D435_COLOR_RESOLUTION[1]}  "
        f"FOV {D435_COLOR_HFOV_DEG:.2f}° × {D435_COLOR_VFOV_DEG:.2f}°"
    )
    print(
        f"  - 内参: fx={D435_COLOR_FX:.4f} fy={D435_COLOR_FY:.4f} "
        f"cx={D435_COLOR_CX:.4f} cy={D435_COLOR_CY:.4f}"
    )
    print("\n控制说明:")
    print("  - 空格: 暂停/继续（暂停时停止录制）")
    print("  - R: 重置场景并开启新 LeRobot session")
    print("  - Tab: 切换 Viewer 到机器人视角 / 自由相机")
    print("  - W/S: 腰部旋转 (+/-)")
    print("  - 右臂(5DOF): 肩 pitch/roll/yaw + 肘 pitch + 腕 roll")
    print("  - Q/E: 右臂朝杯子靠近/远离（肩 roll/pitch/yaw + 肘联动）")
    print("  - ↑/↓: 右臂抬起/下压（主要改高度）")
    print("  - T/G: 肩 pitch 微调 (+/-)")
    print("  - A/D: 肩 roll 展收 (+/-)")
    print("  - Z/X: 肩 yaw 旋转 (+/-)")
    print("  - C/V: 肘伸直/弯曲 (+/-)")
    print("  - B/N: 腕 roll (+/-)")
    print("  - F/H: 张开/闭合任务夹爪（闭合且夹住杯子后可抬起）")
    print("  - ESC: 关闭 viewer")
    print("\n数据录制:")
    print(f"  - head_camera RGB {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {RECORD_FPS}fps")
    print(f"  - observation.state: {len(PROPRIO_STATE_NAMES)} 维 proprio（不含杯子位姿）")
    print("  - 仿真运行时自动录制 LeRobot dataset（parquet + mp4）")
    print(f"  - 输出目录: {SESSIONS_DATA_DIR}/pickcup_YYYYMMDD_HHMMSS/")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="键盘 teleop 抓杯仿真（macOS 请用 mjpython）")
    add_cup_cli_argument(parser)
    args = parser.parse_args(argv)
    scene = SceneLayout.default(cup_variant=args.cup)
    model = mujoco.MjModel.from_xml_string(build_xml(scene))
    data = mujoco.MjData(model)
    actuator_indices = get_actuator_indices(model)
    robot_ctrl = INITIAL_QPOS.copy()
    teleop_geoms, obstacle_geoms = build_collision_geom_sets(model)
    gripper_geoms = build_named_geom_set(model, GRIPPER_GEOM_NAMES)
    cup_geoms = build_named_geom_set(model, CUP_GEOM_NAMES)
    grasp = GraspState()
    renderer = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    recorder = PickcupLeRobotRecorder(
        SESSIONS_DATA_DIR,
        PROPRIO_STATE_NAMES,
        ACTION_NAMES,
        task=DEFAULT_TASK,
    )

    is_paused = False
    should_exit = False
    next_record_time = 0.0
    pending_actions: deque[tuple[str, str, float]] = deque()
    use_robot_camera_view = False
    viewer_handle: mujoco.viewer.Handle | None = None
    next_preview_wall = 0.0

    reset_scene(model, data, robot_ctrl, actuator_indices, scene=scene)
    session_root = recorder.start_session()
    print_help(cup_variant=scene.cup_variant)
    print(f"\n本次 LeRobot session: {session_root}")
    print("提示: PICKCUP_HEAD_CAMERA_PREVIEW=1 可同时显示 3D 与独立机器人视角；")
    print("      若看不到 3D 窗口，请用 Cmd+Tab 切到 mjpython。")

    def key_callback(key: int | str) -> None:
        nonlocal is_paused, should_exit, use_robot_camera_view

        key = resolve_key(key)

        if key == glfw.KEY_SPACE:
            is_paused = not is_paused
            status = "暂停" if is_paused else "继续"
            print(f">>> 仿真已{status}")
            return

        if key == glfw.KEY_ESCAPE:
            should_exit = True
            print(">>> 正在退出 viewer")
            return

        if key == glfw.KEY_TAB:
            use_robot_camera_view = not use_robot_camera_view
            if viewer_handle is not None:
                set_viewer_robot_camera(viewer_handle, model, use_robot_camera_view)
            mode = "机器人视角 (D435i)" if use_robot_camera_view else "自由相机"
            print(f">>> Viewer 已切换: {mode}")
            return

        if key in (glfw.KEY_R, glfw.KEY_BACKSPACE):
            pending_actions.append(("reset", "", 0.0))
            return

        if key == glfw.KEY_F:
            pending_actions.append(("gripper_open", "", 0.0))
            return

        if key == glfw.KEY_H:
            pending_actions.append(("gripper_close", "", 0.0))
            return

        if key == glfw.KEY_Q:
            pending_actions.append(("reach_forward", "", 0.0))
            return

        if key == glfw.KEY_E:
            pending_actions.append(("reach_backward", "", 0.0))
            return

        if key == glfw.KEY_UP:
            pending_actions.append(("reach_up", "", 0.0))
            return

        if key == glfw.KEY_DOWN:
            pending_actions.append(("reach_down", "", 0.0))
            return

        if key in KEY_BINDINGS:
            joint_name, delta = KEY_BINDINGS[key]
            pending_actions.append(("joint", joint_name, delta))

    def process_pending_actions() -> None:
        nonlocal next_record_time

        while pending_actions:
            action, joint_name, delta = pending_actions.popleft()

            if action == "reset":
                recorder.restart_session()
                reset_scene(model, data, robot_ctrl, actuator_indices, scene=scene)
                grasp.reset()
                next_record_time = 0.0
                print(">>> 场景已重置，已开启新 LeRobot session")
                continue

            if action == "gripper_open":
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    GRIPPER_OPEN,
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(">>> 任务夹爪已张开")
                else:
                    print(">>> 任务夹爪无法继续张开（碰撞限制）")
                continue

            if action == "gripper_close":
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    GRIPPER_CLOSED,
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(">>> 任务夹爪已闭合")
                else:
                    print(">>> 任务夹爪无法继续闭合（碰撞限制）")
                continue

            if action == "reach_forward":
                updates = {
                    joint: robot_ctrl[joint] + delta
                    for joint, delta in REACH_FORWARD_DELTAS.items()
                }
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    updates,
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(">>> 右臂前伸")
                else:
                    print(">>> 右臂已达碰撞极限，无法继续前伸")
                continue

            if action == "reach_backward":
                updates = {
                    joint: robot_ctrl[joint] + delta
                    for joint, delta in REACH_BACKWARD_DELTAS.items()
                }
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    updates,
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(">>> 右臂后收")
                else:
                    print(">>> 右臂已达碰撞极限，无法继续后收")
                continue

            if action == "reach_up":
                updates = {
                    joint: robot_ctrl[joint] + delta
                    for joint, delta in REACH_UP_DELTAS.items()
                }
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    updates,
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(">>> 右臂抬起")
                else:
                    print(">>> 右臂已达碰撞极限，无法继续抬起")
                continue

            if action == "reach_down":
                updates = {
                    joint: robot_ctrl[joint] + delta
                    for joint, delta in REACH_DOWN_DELTAS.items()
                }
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    updates,
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(">>> 右臂下压")
                else:
                    print(">>> 右臂已达碰撞极限，无法继续下压")
                continue

            if action == "joint":
                proposed = robot_ctrl[joint_name] + delta
                if try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    {joint_name: proposed},
                    teleop_geoms,
                    obstacle_geoms,
                ):
                    print(f">>> {joint_name}: {robot_ctrl[joint_name]:.2f} rad")
                else:
                    print(f">>> {joint_name} 已达碰撞极限，无法继续移动")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_right_ui=False,
    ) as viewer:
        viewer_handle = viewer
        set_viewer_robot_camera(viewer, model, use_robot_camera_view)
        # OpenCV GUI 在普通 Python 子进程运行，避免与 mjpython 的 GLFW/Cocoa 主线程冲突。
        preview_inited = False
        try:
            while viewer.is_running() and not should_exit:
                loop_start = time.time()
                preview_rgb: np.ndarray | None = None

                with viewer.lock():
                    process_pending_actions()
                    apply_hold_ctrl(model, data, robot_ctrl, actuator_indices)
                    apply_kinematic_teleop(model, data, robot_ctrl)

                    if not is_paused:
                        mujoco.mj_step(model, data)
                        apply_kinematic_teleop(model, data, robot_ctrl)
                        sanitize_cup_physics(model, data, grasp_active=grasp.active)
                        update_grasp(
                            model,
                            data,
                            robot_ctrl,
                            grasp,
                            gripper_geoms,
                            cup_geoms,
                        )
                        mujoco.mj_forward(model, data)
                        next_record_time, preview_rgb = maybe_record_frame(
                            model,
                            data,
                            robot_ctrl,
                            renderer,
                            recorder,
                            next_record_time,
                        )

                    _configure_viewer(viewer)
                    set_viewer_robot_camera(viewer, model, use_robot_camera_view)

                viewer.sync()

                if not preview_inited:
                    init_head_camera_preview(enable_by_default=False)
                    preview_inited = True

                # 机器人视角窗口：优先用本帧录制 RGB；否则按预览帧率单独渲染。
                if _head_camera_preview_enabled:
                    now = time.time()
                    if preview_rgb is not None:
                        show_head_camera_preview(preview_rgb)
                        next_preview_wall = now + RECORD_PERIOD
                    elif now >= next_preview_wall:
                        show_head_camera_preview(render_head_camera(renderer, data))
                        next_preview_wall = now + RECORD_PERIOD

                with viewer.lock():
                    _configure_viewer(viewer)
                    set_viewer_robot_camera(viewer, model, use_robot_camera_view)

                sleep_time = model.opt.timestep - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n\n用户中断")
        finally:
            print("\n\n正在保存 LeRobot dataset...")
            recorder.close_session()
            close_head_camera_preview()

    print("\n程序退出")


if __name__ == "__main__":
    main()
