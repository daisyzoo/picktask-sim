"""G1-23 (mode_10) URDF → MJCF 挂载：固定 pelvis、头相机、全 23DOF 执行器。

自包含实现，禁止 import picktask。
"""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from paths import (
    G1_COMPILED_MJCF_CACHE,
    G1_DESCRIPTION_DIR,
    G1_URDF_PATH,
)

HEAD_CAMERA_NAME = "head_camera"

# G1-23 官方 d435_link 相对 torso_link；含 ROS→MuJoCo 相机轴转换。
D435_TORSO_POSITION = (0.0576235, 0.01753, 0.42987)
D435_MUJOCO_QUATERNION = (
    0.65925248,
    0.25570719,
    -0.25570719,
    -0.65925248,
)
D435_COLOR_RESOLUTION = (640, 480)
D435_COLOR_FX = 605.5006
D435_COLOR_FY = 605.4136
D435_COLOR_CX = 326.1470
D435_COLOR_CY = 249.5715
D435_COLOR_FOCAL_PIXEL = (D435_COLOR_FX, D435_COLOR_FY)
D435_COLOR_PRINCIPAL_PIXEL = (
    D435_COLOR_CX - D435_COLOR_RESOLUTION[0] / 2,
    D435_COLOR_RESOLUTION[1] / 2 - D435_COLOR_CY,
)
D435_COLOR_SENSOR_SIZE = (6.4, 4.8)

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

# home_scene：G1-23 全部关节可控（与 picktask「仅腰+臂 teleop、腿 hold」解耦）。
ACTUATED_JOINTS = list(G1_JOINTS)
assert len(ACTUATED_JOINTS) == 23

# 兼容旧命名：全自由后不再区分 teleop/hold。
TELEOP_JOINTS = list(ACTUATED_JOINTS)
HOLD_JOINTS = list(ACTUATED_JOINTS)

ACTUATOR_KP = 80.0
ACTUATOR_FORCE_RANGE = (-80.0, 80.0)

# G1 站立 pelvis 高度；XY 由场景布局覆盖。
DEFAULT_ROBOT_POSITION = (0.0, 1.20, 0.793)
DEFAULT_ROBOT_YAW = math.pi / 2


def fmt(values: Sequence[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def preprocess_g1_urdf() -> Path:
    """修正官方 URDF meshdir 双重 meshes 问题，不改原文件。"""
    if not G1_URDF_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 G1 URDF: {G1_URDF_PATH}\n"
            "请确认仓库根下存在 unitree_ros/robots/g1_description/"
        )
    text = G1_URDF_PATH.read_text()
    text = text.replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        '<compiler discardvisual="false"/>',
    )
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix="home_scene_g1_preprocessed_",
        dir=G1_DESCRIPTION_DIR,
        delete=False,
    )
    with temp:
        temp.write(text)
    return Path(temp.name)


def compile_g1_urdf_to_mjcf() -> ET.Element:
    """官方编译器转 MJCF；结果缓存到 home_scene/assets/cache。"""
    if (
        G1_COMPILED_MJCF_CACHE.exists()
        and G1_COMPILED_MJCF_CACHE.stat().st_mtime >= G1_URDF_PATH.stat().st_mtime
    ):
        return ET.parse(G1_COMPILED_MJCF_CACHE).getroot()

    preprocessed = preprocess_g1_urdf()
    try:
        robot_model = mujoco.MjModel.from_xml_path(str(preprocessed))
        G1_COMPILED_MJCF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(G1_COMPILED_MJCF_CACHE), robot_model)
        root = ET.parse(G1_COMPILED_MJCF_CACHE).getroot()
    finally:
        preprocessed.unlink(missing_ok=True)

    root.set("model", "home_scene_g1_23dof_mode_10")
    return root


def set_joint_dynamics(root: ET.Element) -> None:
    for joint in root.findall(".//joint"):
        name = joint.get("name", "")
        if name not in G1_JOINTS:
            continue
        joint.set("damping", "1.0")
        joint.set("armature", "0.01")
        joint.set("actuatorfrcrange", fmt(ACTUATOR_FORCE_RANGE))


def wrap_robot_in_fixed_pelvis(
    root: ET.Element,
    robot_position: Sequence[float],
    robot_yaw: float,
    robot_nodes: list[ET.Element] | None = None,
) -> ET.Element:
    """仅把机器人节点挂到固定 pelvis 下；不得吞并地板/墙/家具。"""
    worldbody = ensure_child(root, "worldbody")
    if robot_nodes is None:
        robot_nodes = list(worldbody)
    for child in robot_nodes:
        if child in list(worldbody):
            worldbody.remove(child)

    pelvis = ET.Element(
        "body",
        {
            "name": "pelvis",
            "pos": fmt(robot_position),
            "quat": fmt(yaw_to_quat(robot_yaw)),
        },
    )
    for child in robot_nodes:
        pelvis.append(child)
    worldbody.append(pelvis)
    return pelvis


def add_head_camera(root: ET.Element) -> None:
    torso = root.find(".//body[@name='torso_link']")
    if torso is None:
        raise ValueError("找不到 torso_link，无法添加 head_camera")

    d435_body = ET.SubElement(
        torso,
        "body",
        {"name": "d435_link", "pos": fmt(D435_TORSO_POSITION)},
    )
    ET.SubElement(
        d435_body,
        "camera",
        {
            "name": HEAD_CAMERA_NAME,
            "pos": "0 0 0",
            "quat": fmt(D435_MUJOCO_QUATERNION),
            "resolution": " ".join(map(str, D435_COLOR_RESOLUTION)),
            "sensorsize": fmt(D435_COLOR_SENSOR_SIZE),
            "focalpixel": fmt(D435_COLOR_FOCAL_PIXEL),
            "principalpixel": fmt(D435_COLOR_PRINCIPAL_PIXEL),
            "mode": "fixed",
        },
    )


def get_joint_range(root: ET.Element, joint_name: str) -> tuple[float, float]:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"找不到关节: {joint_name}")
    range_text = joint.get("range")
    if not range_text:
        return -3.14, 3.14
    low, high = (float(v) for v in range_text.split()[:2])
    return low, high


def add_full_dof_actuators(root: ET.Element) -> None:
    """为 G1 全部 23 个关节挂 position 执行器（全自由度可控）。"""
    old = root.find("actuator")
    if old is not None:
        root.remove(old)
    actuator = ET.SubElement(root, "actuator")
    for joint_name in ACTUATED_JOINTS:
        low, high = get_joint_range(root, joint_name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_pos",
                "joint": joint_name,
                "kp": f"{ACTUATOR_KP:.6g}",
                "ctrlrange": fmt((low, high)),
                "forcelimited": "true",
                "forcerange": fmt(ACTUATOR_FORCE_RANGE),
            },
        )


# 旧名兼容
add_hold_actuators = add_full_dof_actuators


def sync_g1_ctrl_to_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """把 G1 各关节当前位置写入对应 position 执行器 ctrl，避免被拉回 0 位。"""
    for joint_name in ACTUATED_JOINTS:
        act_name = f"{joint_name}_pos"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if aid < 0 or jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        data.ctrl[aid] = float(data.qpos[qadr])


def sanitize_robot_geoms(root: ET.Element) -> None:
    """去掉与 visual 重复的 collision proxy / 原始圆柱球胶囊碰撞体。"""
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is None:
        return

    for body in (pelvis, *pelvis.iter("body")):
        visual_meshes = {
            geom.get("mesh")
            for geom in body.findall("geom")
            if geom.get("mesh") and geom.get("contype") == "0"
        }
        to_remove: list[ET.Element] = []
        for geom in body.findall("geom"):
            mesh = geom.get("mesh")
            if mesh and geom.get("contype") != "0" and mesh in visual_meshes:
                to_remove.append(geom)
                continue
            geom_type = geom.get("type", "mesh" if mesh else "sphere")
            if geom_type in ("cylinder", "sphere", "capsule"):
                to_remove.append(geom)
        for geom in to_remove:
            body.remove(geom)


def mount_g1(
    root: ET.Element,
    *,
    robot_position: Sequence[float] = DEFAULT_ROBOT_POSITION,
    robot_yaw: float = DEFAULT_ROBOT_YAW,
) -> ET.Element:
    """在已有/空 MJCF 树上挂载 G1：编译 → 固定 pelvis → 相机 → 23DOF 执行器。

    只把 URDF 编译出的机器人节点包进 pelvis；worldbody 里已有的灯光/地板/家具保持为兄弟节点。
    pelvis 仍固定（无浮动基）；关节侧 23 DOF 全部挂 position 执行器可独立控制。
    """
    robot_root = compile_g1_urdf_to_mjcf()
    set_joint_dynamics(robot_root)

    compiler = ensure_child(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str(G1_DESCRIPTION_DIR))

    src_asset = robot_root.find("asset")
    if src_asset is not None:
        dst_asset = ensure_child(root, "asset")
        for child in list(src_asset):
            dst_asset.append(child)

    src_world = robot_root.find("worldbody")
    dst_world = ensure_child(root, "worldbody")
    robot_nodes: list[ET.Element] = []
    if src_world is not None:
        for child in list(src_world):
            dst_world.append(child)
            robot_nodes.append(child)

    pelvis = wrap_robot_in_fixed_pelvis(
        root, robot_position, robot_yaw, robot_nodes=robot_nodes
    )
    sanitize_robot_geoms(root)
    add_head_camera(root)
    add_full_dof_actuators(root)
    return pelvis
