"""G1 拎包场景：保留桌+杯，旁置方凳与托特包（Phase 1 几何近似）。"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from collections import deque
import math
from typing import Sequence

import glfw
import mujoco
import mujoco.viewer
import numpy as np

import pickcup as pc
from lerobot_recorder import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PickcupLeRobotRecorder,
)
from paths import LIFTBAG_SESSIONS_DATA_DIR, SESSIONS_DATA_DIR

# ==================== 凳 / 包尺寸（liftbagassets information.txt）====================
STOOL_HEIGHT = 0.48
STOOL_HALF_XY = 0.14  # 28cm × 28cm
STOOL_SEAT_THICKNESS = 0.05  # 座面厚 5cm
STOOL_SEAT_HALF_Z = STOOL_SEAT_THICKNESS / 2
STOOL_SEAT_Z = STOOL_HEIGHT  # 座面顶面

# 桌子 +X 一侧：桌沿到凳沿净空 1m（桌边 x=TABLE_LENGTH/2）
_TABLE_PLUS_X_EDGE = pc.TABLE_LENGTH / 2
_STOOL_CLEARANCE_FROM_TABLE = 1.0
DEFAULT_STOOL_POSITION = (
    _TABLE_PLUS_X_EDGE + _STOOL_CLEARANCE_FROM_TABLE + STOOL_HALF_XY,
    -0.08,
    0.0,
)

BAG_LENGTH = 0.32
BAG_DEPTH = 0.14
BAG_HEIGHT = 0.28
BAG_MASS = 0.30
BAG_HALF = (BAG_LENGTH / 2, BAG_DEPTH / 2, BAG_HEIGHT / 2)
BAG_TOP_HALF_X = BAG_HALF[0]
# 托特包横向不收腰，前后包面从底部开始沿整段平滑曲线靠拢；顶部保留 1cm 开口。
BAG_TOP_HALF_Y = 0.005
BAG_GATHER_START = 0.0
BAG_PROFILE_SEGMENTS = 12

HANDLE_STRAP_WIDTH = 0.012
HANDLE_STRAP_THICKNESS = 0.0015
HANDLE_ATTACH_X = 0.0875
HANDLE_ATTACH_Z = 0.095
HANDLE_ARCH_BASE_Z = BAG_HEIGHT / 2 + 0.006
HANDLE_TOP_Z = 0.32
HANDLE_ARCH_SEGMENTS = 12
HANDLE_TOP_GAP_HALF = HANDLE_STRAP_THICKNESS / 2
HANDLE_PATCH_WIDTH = 0.018
HANDLE_PATCH_HEIGHT = 0.050
BAG_TRIM_WIDTH = 0.008
BAG_TRIM_THICKNESS = 0.0015

LIFT_HEIGHT_MIN = 0.12
HOLD_TIME_S = 1.5

PICKCUP_MODE = "pickcup"
LIFTBAG_MODE = "liftbag"
PICKCUP_TASK = pc.DEFAULT_TASK
LIFTBAG_TASK = "lift the bag"
LIFTBAG_OBSTACLE_BODY_NAMES = frozenset(pc.OBSTACLE_BODY_NAMES | {"stool"})

# 拎包模式下把固定骨盆放到凳子南侧，保持与抓杯模式相同的朝向和安全净空。
LIFTBAG_ROBOT_POSITION = (
    DEFAULT_STOOL_POSITION[0],
    DEFAULT_STOOL_POSITION[1] - STOOL_HALF_XY - pc.ROBOT_TABLE_STANDOFF,
    pc.DEFAULT_ROBOT_POSITION[2],
)

BAG_HANDLE_GEOM_NAMES = frozenset(
    f"bag_handle_{side}_{segment}"
    for side in ("front", "back")
    for segment in (
        "left_stem",
        *(f"arch_{index:02d}" for index in range(HANDLE_ARCH_SEGMENTS)),
        "right_stem",
    )
)


def _bag_profile_rings() -> list[tuple[float, float, float]]:
    """返回 (z, half_x, half_y) 环；前后包面从底到顶以 smoothstep 连续收拢。"""
    rings: list[tuple[float, float, float]] = []
    for index in range(BAG_PROFILE_SEGMENTS + 1):
        height_ratio = index / BAG_PROFILE_SEGMENTS
        gather_ratio = float(
            np.clip(
                (height_ratio - BAG_GATHER_START) / (1.0 - BAG_GATHER_START),
                0.0,
                1.0,
            )
        )
        smooth_gather = gather_ratio * gather_ratio * (3.0 - 2.0 * gather_ratio)
        z = -BAG_HALF[2] + BAG_HEIGHT * height_ratio
        half_x = BAG_HALF[0] + (BAG_TOP_HALF_X - BAG_HALF[0]) * smooth_gather
        half_y = BAG_HALF[1] + (BAG_TOP_HALF_Y - BAG_HALF[1]) * smooth_gather
        rings.append((z, half_x, half_y))
    return rings


def _bag_body_mesh_geometry(*, open_top: bool) -> tuple[str, str]:
    rings = _bag_profile_rings()
    vertices: list[tuple[float, float, float]] = []
    for z, half_x, half_y in rings:
        vertices.extend(
            [
                (-half_x, -half_y, z),
                (half_x, -half_y, z),
                (half_x, half_y, z),
                (-half_x, half_y, z),
            ]
        )

    faces: list[tuple[int, int, int]] = [(0, 2, 1), (0, 3, 2)]
    for index in range(len(rings) - 1):
        lower = index * 4
        upper = lower + 4
        faces.extend(
            [
                (lower, lower + 1, upper + 1),
                (lower, upper + 1, upper),
                (lower + 1, lower + 2, upper + 2),
                (lower + 1, upper + 2, upper + 1),
                (lower + 2, lower + 3, upper + 3),
                (lower + 2, upper + 3, upper + 2),
                (lower + 3, lower, upper),
                (lower + 3, upper, upper + 3),
            ]
        )
    if not open_top:
        top = (len(rings) - 1) * 4
        faces.extend(((top, top + 1, top + 2), (top, top + 2, top + 3)))
    return (
        pc._fmt(tuple(value for vertex in vertices for value in vertex)),
        " ".join(str(value) for face in faces for value in face),
    )


def _handle_centerline(sign: float) -> np.ndarray:
    """一条提带从左根部到右根部的中心线。"""
    arch_y = BAG_TOP_HALF_Y + 0.005
    attach_y = _bag_surface_half_y(HANDLE_ATTACH_Z) + BAG_TRIM_THICKNESS / 2
    return np.asarray(
        [
            (-HANDLE_ATTACH_X, sign * attach_y, HANDLE_ATTACH_Z),
            (-0.0875, sign * arch_y, HANDLE_ARCH_BASE_Z),
            (-0.0850, sign * 0.024, 0.190),
            (-0.0775, sign * 0.015, 0.230),
            (-0.0600, sign * 0.008, 0.270),
            (-0.0400, sign * HANDLE_TOP_GAP_HALF, 0.300),
            (-0.0200, sign * HANDLE_TOP_GAP_HALF, 0.315),
            (0.0000, sign * HANDLE_TOP_GAP_HALF, HANDLE_TOP_Z),
            (0.0200, sign * HANDLE_TOP_GAP_HALF, 0.315),
            (0.0400, sign * HANDLE_TOP_GAP_HALF, 0.300),
            (0.0600, sign * 0.008, 0.270),
            (0.0775, sign * 0.015, 0.230),
            (0.0850, sign * 0.024, 0.190),
            (0.0875, sign * arch_y, HANDLE_ARCH_BASE_Z),
            (HANDLE_ATTACH_X, sign * attach_y, HANDLE_ATTACH_Z),
        ],
        dtype=np.float64,
    )


def _smooth_polyline(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """用 Chaikin 切角生成不越出原轮廓的平滑中心线。"""
    smoothed = points
    for _ in range(iterations):
        result = [smoothed[0]]
        for start, finish in zip(smoothed[:-1], smoothed[1:]):
            result.extend((0.75 * start + 0.25 * finish, 0.25 * start + 0.75 * finish))
        result.append(smoothed[-1])
        smoothed = np.asarray(result, dtype=np.float64)
    return smoothed


def _flat_strap_mesh(sign: float) -> tuple[str, str]:
    """生成带厚度的连续扁平提带 mesh，避免分段 box 的锯齿接缝。"""
    points = _smooth_polyline(_handle_centerline(sign))
    sections: list[np.ndarray] = []
    for index, center in enumerate(points):
        if index == 0:
            tangent = points[1] - center
        elif index == len(points) - 1:
            tangent = center - points[index - 1]
        else:
            tangent = points[index + 1] - points[index - 1]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)

        normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        normal -= float(normal @ tangent) * tangent
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        width_axis = np.cross(normal, tangent)
        width_axis /= max(float(np.linalg.norm(width_axis)), 1e-12)
        half_width = width_axis * (HANDLE_STRAP_WIDTH / 2)
        half_thickness = normal * (HANDLE_STRAP_THICKNESS / 2)
        sections.append(
            np.asarray(
                [
                    center - half_width - half_thickness,
                    center + half_width - half_thickness,
                    center + half_width + half_thickness,
                    center - half_width + half_thickness,
                ]
            )
        )

    vertices = np.concatenate(sections, axis=0)
    faces: list[tuple[int, int, int]] = [(0, 2, 1), (0, 3, 2)]
    for index in range(len(sections) - 1):
        current = index * 4
        following = current + 4
        faces.extend(
            [
                (current, following + 1, following),
                (current, current + 1, following + 1),
                (current + 1, following + 2, following + 1),
                (current + 1, current + 2, following + 2),
                (current + 2, following + 3, following + 2),
                (current + 2, current + 3, following + 3),
                (current + 3, following, following + 3),
                (current + 3, current, following),
            ]
        )
    final = (len(sections) - 1) * 4
    faces.extend(((final, final + 1, final + 2), (final, final + 2, final + 3)))
    return (
        pc._fmt(tuple(float(value) for value in vertices.reshape(-1))),
        " ".join(str(value) for face in faces for value in face),
    )


def _default_bag_position() -> np.ndarray:
    sx, sy, _ = DEFAULT_STOOL_POSITION
    return np.array(
        [sx, sy, STOOL_SEAT_Z + BAG_HALF[2] + 0.001],
        dtype=np.float64,
    )


def _add_liftbag_materials(root: ET.Element) -> None:
    asset = pc._ensure_child(root, "asset")
    for name, rgba in {
        "stool_black": "0.08 0.08 0.09 1",
        "bag_iridescent": "0.42 0.86 0.91 0.48",
        "bag_front_print": "0.72 0.88 0.90 0.42",
        "bag_pink": "0.72 0.28 0.58 0.92",
        "bag_trim_pink": "0.72 0.28 0.58 1",
        "print_peach": "0.98 0.70 0.60 0.9",
        "print_yellow": "0.96 0.86 0.42 0.9",
        "print_blue": "0.47 0.78 0.92 0.9",
        "print_white": "0.97 0.97 0.92 0.95",
        "print_black": "0.04 0.04 0.05 1",
    }.items():
        if asset.find(f"./material[@name='{name}']") is None:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    if asset.find("./mesh[@name='bag_body_mesh']") is None:
        vertices, faces = _bag_body_mesh_geometry(open_top=False)
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "bag_body_mesh",
                "vertex": vertices,
                "face": faces,
            },
        )
    if asset.find("./mesh[@name='bag_body_visual_mesh']") is None:
        vertices, faces = _bag_body_mesh_geometry(open_top=True)
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "bag_body_visual_mesh",
                "vertex": vertices,
                "face": faces,
            },
        )
    for side, sign in (("front", -1.0), ("back", 1.0)):
        mesh_name = f"bag_handle_{side}_mesh"
        if asset.find(f"./mesh[@name='{mesh_name}']") is None:
            vertices, faces = _flat_strap_mesh(sign)
            ET.SubElement(
                asset,
                "mesh",
                {"name": mesh_name, "vertex": vertices, "face": faces},
            )


def _quat_from_z_to(vector: tuple[float, float, float]) -> str:
    """返回把局部 Z 轴旋到 vector 的 wxyz 四元数。"""
    direction = np.asarray(vector, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    dot = float(np.clip(direction[2], -1.0, 1.0))
    angle = math.acos(dot)
    axis = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return "1 0 0 0" if dot >= 0 else "0 1 0 0"
    axis /= axis_norm
    quat = np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2)))
    return pc._fmt(tuple(float(value) for value in quat))


def _flat_strap_segment_attrs(
    start: tuple[float, float, float],
    finish: tuple[float, float, float],
    *,
    width: float = HANDLE_STRAP_WIDTH,
    thickness: float = HANDLE_STRAP_THICKNESS,
) -> dict[str, str]:
    """生成沿线段布置的薄矩形饰条。"""
    start_array = np.asarray(start, dtype=np.float64)
    finish_array = np.asarray(finish, dtype=np.float64)
    tangent = finish_array - start_array
    length = float(np.linalg.norm(tangent))
    tangent /= max(length, 1e-12)

    # 先用世界 Y 轴作为带材表面法向，再投影到垂直于切线的平面。
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal -= float(normal @ tangent) * tangent
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    width_axis = np.cross(normal, tangent)
    width_axis /= max(float(np.linalg.norm(width_axis)), 1e-12)
    rotation = np.column_stack((width_axis, normal, tangent))
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))

    midpoint = 0.5 * (start_array + finish_array)
    return {
        "type": "box",
        "size": pc._fmt((width / 2, thickness / 2, length / 2)),
        "pos": pc._fmt(tuple(float(value) for value in midpoint)),
        "quat": pc._fmt(tuple(float(value) for value in quat)),
    }


def _bag_surface_half_y(z: float) -> float:
    """包体侧面在局部高度 z 处的半厚度。"""
    height_ratio = float(np.clip((z + BAG_HALF[2]) / BAG_HEIGHT, 0.0, 1.0))
    gather_ratio = float(
        np.clip(
            (height_ratio - BAG_GATHER_START) / (1.0 - BAG_GATHER_START),
            0.0,
            1.0,
        )
    )
    smooth_gather = gather_ratio * gather_ratio * (3.0 - 2.0 * gather_ratio)
    return BAG_HALF[1] + (BAG_TOP_HALF_Y - BAG_HALF[1]) * smooth_gather


def _add_random_bag_prints(
    bag: ET.Element,
    rng: np.random.Generator,
    visual_attrs: dict[str, str],
) -> None:
    """在包体前后两面分别生成一组随机彩色卡通图案。"""
    palette = ("print_peach", "print_yellow", "print_blue", "print_white")
    slots = (
        (-0.095, 0.060),
        (-0.075, -0.055),
        (0.005, 0.035),
        (0.070, -0.060),
        (0.095, 0.055),
    )

    for side, sign in (("front", -1.0), ("back", 1.0)):
        cylinder_to_face = (
            "0.707107 0.707107 0 0" if sign < 0 else "0.707107 -0.707107 0 0"
        )
        slot_indices = rng.permutation(len(slots))[: int(rng.integers(3, 6))]
        for pattern_index, slot_index in enumerate(slot_indices, start=1):
            base_x, base_z = slots[int(slot_index)]
            x = float(base_x + rng.uniform(-0.010, 0.010))
            z = float(base_z + rng.uniform(-0.010, 0.010))
            radius = float(rng.uniform(0.024, 0.040))
            face_y = sign * (_bag_surface_half_y(z) + 0.002)
            material = palette[int(rng.integers(0, len(palette)))]
            name = f"bag_print_{side}_{pattern_index}"

            ET.SubElement(
                bag,
                "geom",
                {
                    "name": name,
                    "type": "cylinder",
                    "size": pc._fmt((radius, 0.0015)),
                    "pos": pc._fmt((x, face_y, z)),
                    "quat": cylinder_to_face,
                    "material": material,
                    **visual_attrs,
                },
            )

            eye_z = z + radius * float(rng.uniform(0.05, 0.18))
            eye_spacing = radius * float(rng.uniform(0.22, 0.32))
            for eye_index, eye_x in enumerate((x - eye_spacing, x + eye_spacing), start=1):
                ET.SubElement(
                    bag,
                    "geom",
                    {
                        "name": f"{name}_eye{eye_index}",
                        "type": "cylinder",
                        "size": pc._fmt((radius * 0.105, 0.0018)),
                        "pos": pc._fmt((eye_x, face_y + sign * 0.003, eye_z)),
                        "quat": cylinder_to_face,
                        "material": "print_black",
                        **visual_attrs,
                    },
                )

        # 每一面再加少量随机彩点，使前后印花明显不同。
        for dot_index in range(6):
            x = float(rng.uniform(-0.125, 0.125))
            z = float(rng.uniform(-0.100, 0.105))
            radius = float(rng.uniform(0.003, 0.007))
            face_y = sign * (_bag_surface_half_y(z) + 0.0025)
            ET.SubElement(
                bag,
                "geom",
                {
                    "name": f"bag_print_{side}_dot{dot_index + 1}",
                    "type": "cylinder",
                    "size": pc._fmt((radius, 0.0012)),
                    "pos": pc._fmt((x, face_y, z)),
                    "quat": cylinder_to_face,
                    "material": palette[int(rng.integers(0, len(palette)))],
                    **visual_attrs,
                },
            )


def _add_stool_and_bag(
    root: ET.Element,
    rng: np.random.Generator | None = None,
) -> None:
    """按参考图搭建圆角带孔塑料凳与直立透明托特包。"""
    if rng is None:
        rng = np.random.default_rng()
    worldbody = pc._ensure_child(root, "worldbody")
    sx, sy, _sz = DEFAULT_STOOL_POSITION
    seat_center_z = STOOL_SEAT_Z - STOOL_SEAT_HALF_Z
    # 腿顶贴座面底（座面厚 5cm）
    leg_top_z = STOOL_SEAT_Z - STOOL_SEAT_THICKNESS

    stool = ET.SubElement(
        worldbody,
        "body",
        {"name": "stool", "pos": pc._fmt((sx, sy, 0.0))},
    )
    # 交叉的四块板留下 2.8cm 中央方孔；四角圆柱形成约 2cm 圆角。
    hole_half = 0.014
    corner_radius = 0.02
    inner_extent = STOOL_HALF_XY - corner_radius
    seat_pieces = [
        ("left", (-(STOOL_HALF_XY + hole_half) / 2, 0.0), ((STOOL_HALF_XY - hole_half) / 2, inner_extent)),
        ("right", ((STOOL_HALF_XY + hole_half) / 2, 0.0), ((STOOL_HALF_XY - hole_half) / 2, inner_extent)),
        ("back", (0.0, -(STOOL_HALF_XY + hole_half) / 2), (inner_extent, (STOOL_HALF_XY - hole_half) / 2)),
        ("front", (0.0, (STOOL_HALF_XY + hole_half) / 2), (inner_extent, (STOOL_HALF_XY - hole_half) / 2)),
    ]
    for name, (px, py), (hx, hy) in seat_pieces:
        ET.SubElement(
            stool,
            "geom",
            {
                "name": f"stool_seat_{name}",
                "type": "box",
                "size": pc._fmt((hx, hy, STOOL_SEAT_HALF_Z)),
                "pos": pc._fmt((px, py, seat_center_z)),
                "material": "stool_black",
                **pc._contact_attrs(),
            },
        )
    for i, (cx, cy) in enumerate(
        [
            (-inner_extent, -inner_extent),
            (inner_extent, -inner_extent),
            (-inner_extent, inner_extent),
            (inner_extent, inner_extent),
        ],
        start=1,
    ):
        ET.SubElement(
            stool,
            "geom",
            {
                "name": f"stool_seat_corner{i}",
                "type": "cylinder",
                "size": pc._fmt((corner_radius, STOOL_SEAT_HALF_Z)),
                "pos": pc._fmt((cx, cy, seat_center_z)),
                "material": "stool_black",
                **pc._contact_attrs(),
            },
        )

    # 参考图中四腿为细长矩形塑料腿，由座面角向外轻微张开。
    leg_top = STOOL_HALF_XY - 0.025
    leg_bottom = STOOL_HALF_XY - 0.005
    for i, (tx, ty, bx, by) in enumerate(
        [
            (-leg_top, -leg_top, -leg_bottom, -leg_bottom),
            (leg_top, -leg_top, leg_bottom, -leg_bottom),
            (-leg_top, leg_top, -leg_bottom, leg_bottom),
            (leg_top, leg_top, leg_bottom, leg_bottom),
        ],
        start=1,
    ):
        start = np.array((bx, by, 0.015), dtype=np.float64)
        finish = np.array((tx, ty, leg_top_z), dtype=np.float64)
        midpoint = 0.5 * (start + finish)
        vector = finish - start
        ET.SubElement(
            stool,
            "geom",
            {
                "name": f"stool_leg{i}",
                "type": "box",
                "size": pc._fmt((0.015, 0.011, float(np.linalg.norm(vector)) / 2)),
                "pos": pc._fmt(tuple(float(value) for value in midpoint)),
                "quat": _quat_from_z_to(tuple(float(value) for value in vector)),
                "material": "stool_black",
                **pc._contact_attrs(),
            },
        )

    bag_pos = _default_bag_position()
    bag = ET.SubElement(
        worldbody,
        "body",
        {"name": "bag", "pos": pc._fmt(tuple(float(v) for v in bag_pos))},
    )
    ET.SubElement(
        bag,
        "joint",
        {"name": "bag_free", "type": "free", "damping": "0.2", "armature": "0.001"},
    )
    # 封闭 mesh 只负责质量与碰撞；开口 mesh 负责透明托特包外观。
    ET.SubElement(
        bag,
        "geom",
        {
            "name": "bag_body",
            "type": "mesh",
            "mesh": "bag_body_mesh",
            "rgba": "0 0 0 0",
            "mass": f"{BAG_MASS:.6g}",
            **pc._contact_attrs(),
        },
    )

    visual_attrs = {"contype": "0", "conaffinity": "0", "mass": "0"}
    ET.SubElement(
        bag,
        "geom",
        {
            "name": "bag_body_visual",
            "type": "mesh",
            "mesh": "bag_body_visual_mesh",
            "material": "bag_iridescent",
            **visual_attrs,
        },
    )
    # 参考图的包边是压扁的粉色饰条，不是圆管。
    xh, yh, zh = BAG_HALF
    trim_segments: list[tuple[str, tuple[float, ...]]] = []
    profile_rings = _bag_profile_rings()
    for side, sign in (("front", -1.0), ("back", 1.0)):
        for edge_name, x_sign in (("left", -1.0), ("right", 1.0)):
            for index, (lower, upper) in enumerate(
                zip(profile_rings[:-1], profile_rings[1:])
            ):
                lower_z, lower_x, lower_y = lower
                upper_z, upper_x, upper_y = upper
                trim_segments.append(
                    (
                        f"{side}_{edge_name}_{index:02d}",
                        (
                            x_sign * lower_x,
                            sign * (lower_y + BAG_TRIM_THICKNESS / 2),
                            lower_z,
                            x_sign * upper_x,
                            sign * (upper_y + BAG_TRIM_THICKNESS / 2),
                            upper_z,
                        ),
                    )
                )
        bottom_z, bottom_x, bottom_half_y = profile_rings[0]
        bottom_y = sign * (bottom_half_y + BAG_TRIM_THICKNESS / 2)
        trim_segments.append(
            (
                f"{side}_bottom",
                (-bottom_x, bottom_y, bottom_z, bottom_x, bottom_y, bottom_z),
            )
        )
        # 前后各一条扁平包口边，二者之间保留真实开口；不再添加中央圆柱。
        top_z, top_x, top_half_y = profile_rings[-1]
        top_y = sign * (top_half_y + BAG_TRIM_THICKNESS / 2)
        trim_segments.append(
            (
                f"{side}_top",
                (
                    -top_x,
                    top_y,
                    top_z,
                    top_x,
                    top_y,
                    top_z,
                ),
            )
        )
    for name, line in trim_segments:
        ET.SubElement(
            bag,
            "geom",
            {
                "name": f"bag_trim_{name}",
                **_flat_strap_segment_attrs(
                    tuple(float(value) for value in line[:3]),
                    tuple(float(value) for value in line[3:]),
                    width=BAG_TRIM_WIDTH,
                    thickness=BAG_TRIM_THICKNESS,
                ),
                "material": "bag_trim_pink",
                **visual_attrs,
            },
        )

    _add_random_bag_prints(bag, rng, visual_attrs)

    # 连续 mesh 负责平滑外观；同名分段 box 仅负责抓取碰撞，设为完全透明。
    for side, sign in (("front", -1.0), ("back", 1.0)):
        patch_center_z = zh - HANDLE_PATCH_HEIGHT / 2
        patch_y = sign * (
            _bag_surface_half_y(patch_center_z) + BAG_TRIM_THICKNESS / 2
        )
        for patch_side, x in (("left", -HANDLE_ATTACH_X), ("right", HANDLE_ATTACH_X)):
            ET.SubElement(
                bag,
                "geom",
                {
                    "name": f"bag_handle_{side}_{patch_side}_patch",
                    "type": "box",
                    "size": pc._fmt(
                        (
                            HANDLE_PATCH_WIDTH / 2,
                            BAG_TRIM_THICKNESS / 2,
                            HANDLE_PATCH_HEIGHT / 2,
                        )
                    ),
                    "pos": pc._fmt((x, patch_y, patch_center_z)),
                    "material": "bag_trim_pink",
                    **visual_attrs,
                },
            )
        ET.SubElement(
            bag,
            "geom",
            {
                "name": f"bag_handle_{side}_visual",
                "type": "mesh",
                "mesh": f"bag_handle_{side}_mesh",
                "material": "bag_pink",
                **visual_attrs,
            },
        )
        centerline = _handle_centerline(sign)
        handle_lines = {
            "left_stem": (*centerline[0], *centerline[1]),
            **{
                f"arch_{index:02d}": (*centerline[index + 1], *centerline[index + 2])
                for index in range(HANDLE_ARCH_SEGMENTS)
            },
            "right_stem": (*centerline[-2], *centerline[-1]),
        }
        for segment, line in handle_lines.items():
            ET.SubElement(
                bag,
                "geom",
                {
                    "name": f"bag_handle_{side}_{segment}",
                    **_flat_strap_segment_attrs(
                        tuple(float(value) for value in line[:3]),
                        tuple(float(value) for value in line[3:]),
                    ),
                    "rgba": "0 0 0 0",
                    "mass": "0.002",
                    **pc._contact_attrs(),
                },
            )


def build_xml(scene: pc.SceneLayout | None = None) -> str:
    """桌+杯保留，追加旁置方凳与托特包。"""
    layout = scene or pc.SceneLayout.default()
    variant = pc.resolve_cup_variant(layout.cup_variant)
    root = pc._compile_g1_urdf_to_mjcf()
    root.set("model", "lift_bag_with_g1_23dof_mode_10")
    pc._set_joint_dynamics(root)
    pc._add_scene_assets(root, variant)
    _add_liftbag_materials(root)
    pc._add_world_options(root)
    pelvis = pc._wrap_robot_in_fixed_pelvis(root, layout.robot_position, layout.robot_yaw)
    pc._add_task_gripper(pelvis)
    pc._add_task_world(root, layout.cup_position, variant)
    _add_stool_and_bag(root)
    pc._sanitize_robot_geoms(root)
    pc._add_head_camera(root, layout.robot_position, layout.cup_position, layout.robot_yaw)
    pc._add_position_actuators(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def make_model_data(scene: pc.SceneLayout | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(build_xml(scene))
    return model, mujoco.MjData(model)


def build_liftbag_collision_geom_sets(
    model: mujoco.MjModel,
) -> tuple[frozenset[int], frozenset[int]]:
    body_ids_for = lambda names: {
        pc.get_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names
    }
    teleop_body_ids = body_ids_for(pc.TELEOP_COLLISION_BODIES)
    obstacle_body_ids = body_ids_for(LIFTBAG_OBSTACLE_BODY_NAMES)

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


def get_bag_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    bag_joint_id = pc.get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bag_free")
    qpos_addr = model.jnt_qposadr[bag_joint_id]
    pos = data.qpos[qpos_addr : qpos_addr + 3].copy()
    quat = data.qpos[qpos_addr + 3 : qpos_addr + 7].copy()
    return pos, quat


def set_bag_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    quat: np.ndarray,
) -> None:
    bag_joint_id = pc.get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bag_free")
    qpos_addr = model.jnt_qposadr[bag_joint_id]
    qvel_addr = model.jnt_dofadr[bag_joint_id]
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    if qvel_addr >= 0:
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def set_fixed_robot_pose(
    model: mujoco.MjModel,
    position: tuple[float, float, float],
    yaw: float = pc.DEFAULT_ROBOT_YAW,
) -> None:
    """移动固定 pelvis，用于在两个操作工位之间切换。"""
    pelvis_id = pc.get_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    model.body_pos[pelvis_id] = position
    model.body_quat[pelvis_id] = pc.yaw_to_quat(yaw)


def robot_position_for_mode(mode: str) -> tuple[float, float, float]:
    if mode == PICKCUP_MODE:
        return tuple(pc.DEFAULT_ROBOT_POSITION)
    if mode == LIFTBAG_MODE:
        return LIFTBAG_ROBOT_POSITION
    raise ValueError(f"未知任务模式: {mode}")


def _bag_over_stool(bag_pos: np.ndarray) -> bool:
    sx, sy, _ = DEFAULT_STOOL_POSITION
    return (
        abs(float(bag_pos[0]) - sx) <= STOOL_HALF_XY + 0.05
        and abs(float(bag_pos[1]) - sy) <= STOOL_HALF_XY + 0.05
    )


def sanitize_bag_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    grasp_active: bool = False,
) -> None:
    """未吸附时防止包穿透凳面或地面。"""
    if grasp_active:
        return

    bag_pos, bag_quat = get_bag_pose(model, data)
    bag_joint_id = pc.get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bag_free")
    qvel_addr = model.jnt_dofadr[bag_joint_id]
    if qvel_addr < 0:
        return

    if _bag_over_stool(bag_pos):
        min_z = STOOL_SEAT_Z + BAG_HALF[2] - 0.01
        target_z = STOOL_SEAT_Z + BAG_HALF[2] + 0.001
    else:
        min_z = pc.FLOOR_Z + BAG_HALF[2]
        target_z = min_z

    if bag_pos[2] >= min_z:
        return

    bag_pos = bag_pos.copy()
    bag_pos[2] = target_z
    set_bag_pose(model, data, bag_pos, bag_quat)
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def can_initiate_bag_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    gripper_geoms: frozenset[int],
    handle_geoms: frozenset[int],
) -> bool:
    """夹爪闭合且接触把手（或靠近把手）时可吸附整包。"""
    if not pc.is_gripper_closed(robot_ctrl):
        return False
    if pc.geoms_in_contact(data, gripper_geoms, handle_geoms):
        return True
    gripper_center = pc.get_gripper_center(model, data)
    for name in BAG_HANDLE_GEOM_NAMES:
        gid = pc.get_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if float(np.linalg.norm(data.geom_xpos[gid] - gripper_center)) < 0.06:
            return True
    return False


def update_bag_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    grasp: pc.GraspState,
    gripper_geoms: frozenset[int],
    handle_geoms: frozenset[int],
) -> None:
    """对包 free-joint 做运动学吸附；张开夹爪时释放。"""
    anchor_pos, anchor_mat = pc.get_grip_anchor_pose(model, data)

    if grasp.active:
        if pc.is_gripper_open(robot_ctrl):
            grasp.reset()
            print(">>> 已释放包")
            return

        bag_mat = anchor_mat @ grasp.local_mat
        bag_pos = anchor_pos + anchor_mat @ grasp.local_pos
        bag_quat = np.zeros(4)
        mujoco.mju_mat2Quat(bag_quat, bag_mat.reshape(-1))
        set_bag_pose(model, data, bag_pos, bag_quat)
        return

    if can_initiate_bag_grasp(model, data, robot_ctrl, gripper_geoms, handle_geoms):
        bag_pos, bag_quat = get_bag_pose(model, data)
        bag_mat = np.zeros(9)
        mujoco.mju_quat2Mat(bag_mat, bag_quat)
        grasp.local_pos = anchor_mat.T @ (bag_pos - anchor_pos)
        grasp.local_mat = anchor_mat.T @ bag_mat.reshape(3, 3)
        grasp.active = True
        print(">>> 已抓住包（把手吸附）")


def bag_lift_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    bag_pos, _ = get_bag_pose(model, data)
    rest_z = float(_default_bag_position()[2])
    return float(bag_pos[2] - rest_z)


def reset_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    actuator_indices: dict[str, int],
    *,
    scene: pc.SceneLayout | None = None,
    robot_position: tuple[float, float, float] | None = None,
) -> None:
    if robot_position is not None:
        set_fixed_robot_pose(model, robot_position)
    pc.reset_scene(model, data, robot_ctrl, actuator_indices, scene=scene)
    set_bag_pose(
        model,
        data,
        _default_bag_position(),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    mujoco.mj_forward(model, data)


def maybe_record_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    renderer: mujoco.Renderer,
    recorder: PickcupLeRobotRecorder,
    next_record_time: float,
    task: str = LIFTBAG_TASK,
) -> tuple[float, np.ndarray | None]:
    last_rgb: np.ndarray | None = None
    while data.time + 1e-9 >= next_record_time:
        rgb = pc.render_head_camera(renderer, data)
        last_rgb = rgb
        recorder.add_frame(
            observation_state=pc.collect_proprio_state(model, data),
            action=pc.collect_action_vector(robot_ctrl),
            rgb=rgb,
            task=task,
        )
        next_record_time += pc.RECORD_PERIOD
    return next_record_time, last_rgb


def print_help(current_mode: str, *, cup_variant: str | None = None) -> None:
    variant = pc.resolve_cup_variant(cup_variant)
    variant_label = pc.get_cup_variant_spec(variant)["label"]
    print("=" * 60)
    print("MuJoCo G1 双任务场景（拿杯子 / 拎包）")
    print("=" * 60)
    print("\n场景:")
    print("  - 桌子与杯子：与 pickcup 相同")
    print(f"  - 杯子变体: {variant}（{variant_label}）")
    print(
        f"  - 方凳: {STOOL_HALF_XY * 2 * 100:.0f}cm 边长, 高 {STOOL_HEIGHT * 100:.0f}cm, "
        f"座面厚 {STOOL_SEAT_THICKNESS * 100:.0f}cm；"
        f"相对桌沿净空 {_STOOL_CLEARANCE_FROM_TABLE:.1f}m "
        f"@ ({DEFAULT_STOOL_POSITION[0]:.2f}, {DEFAULT_STOOL_POSITION[1]:.2f})"
    )
    print(
        f"  - 包体: 横向 {BAG_LENGTH * 100:.0f}cm × 高 {BAG_HEIGHT * 100:.0f}cm "
        f"× 厚 {BAG_DEPTH * 100:.0f}cm + 前后双拱形刚体把手（非软带）"
    )
    print(
        f"  - 成功草案: 相对凳面抬升 ≥{LIFT_HEIGHT_MIN * 100:.0f}cm 且保持 ≥{HOLD_TIME_S:.1f}s"
    )
    print("\n任务切换:")
    print("  - 1: 拿杯子（机器人移动到桌边）")
    print("  - 2: 拎包（机器人移动到凳子旁）")
    print(f"  - 当前模式: {'[1] 拿杯子' if current_mode == PICKCUP_MODE else '[2] 拎包'}")
    print("\n控制: F/H 开合夹爪；其余右臂控制与抓杯 teleop 相同")
    print(f"  - 拿杯数据: {SESSIONS_DATA_DIR}/pickcup_YYYYMMDD_HHMMSS/")
    print(f"  - 拎包数据: {LIFTBAG_SESSIONS_DATA_DIR}/liftbag_YYYYMMDD_HHMMSS/")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="键盘 teleop 双任务（拿杯/拎包）")
    pc.add_cup_cli_argument(parser)
    args = parser.parse_args(argv)
    scene = pc.SceneLayout.default(cup_variant=args.cup)
    model = mujoco.MjModel.from_xml_string(build_xml(scene))
    data = mujoco.MjData(model)
    actuator_indices = pc.get_actuator_indices(model)
    robot_ctrl = pc.INITIAL_QPOS.copy()
    teleop_geoms, obstacle_geoms = build_liftbag_collision_geom_sets(model)
    gripper_geoms = pc.build_named_geom_set(model, pc.GRIPPER_GEOM_NAMES)
    handle_geoms = pc.build_named_geom_set(model, BAG_HANDLE_GEOM_NAMES)
    cup_geoms = pc.build_named_geom_set(model, pc.CUP_GEOM_NAMES)
    cup_grasp = pc.GraspState()
    bag_grasp = pc.GraspState()
    renderer = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    recorders = {
        PICKCUP_MODE: PickcupLeRobotRecorder(
            SESSIONS_DATA_DIR,
            pc.PROPRIO_STATE_NAMES,
            pc.ACTION_NAMES,
            task=PICKCUP_TASK,
            session_prefix="pickcup",
        ),
        LIFTBAG_MODE: PickcupLeRobotRecorder(
            LIFTBAG_SESSIONS_DATA_DIR,
            pc.PROPRIO_STATE_NAMES,
            pc.ACTION_NAMES,
            task=LIFTBAG_TASK,
            session_prefix="liftbag",
        ),
    }

    current_mode = LIFTBAG_MODE
    is_paused = False
    should_exit = False
    next_record_time = 0.0
    pending_actions: deque[tuple[str, str, float]] = deque()
    use_robot_camera_view = False
    viewer_handle: mujoco.viewer.Handle | None = None
    next_preview_wall = 0.0
    hold_accum = 0.0
    success_announced = False

    reset_scene(
        model,
        data,
        robot_ctrl,
        actuator_indices,
        scene=scene,
        robot_position=robot_position_for_mode(current_mode),
    )
    session_root = recorders[current_mode].start_session()
    print_help(current_mode, cup_variant=scene.cup_variant)
    print(f"\n本次 LeRobot session: {session_root}")
    print("提示: PICKCUP_HEAD_CAMERA_PREVIEW=1 可同时显示 3D 与独立机器人视角。")

    def key_callback(key: int | str) -> None:
        nonlocal is_paused, should_exit, use_robot_camera_view

        key = pc.resolve_key(key)

        if key == glfw.KEY_SPACE:
            is_paused = not is_paused
            print(f">>> 仿真已{'暂停' if is_paused else '继续'}")
            return
        if key == glfw.KEY_ESCAPE:
            should_exit = True
            print(">>> 正在退出 viewer")
            return
        if key == glfw.KEY_TAB:
            use_robot_camera_view = not use_robot_camera_view
            if viewer_handle is not None:
                pc.set_viewer_robot_camera(viewer_handle, model, use_robot_camera_view)
            mode = "机器人视角 (D435i)" if use_robot_camera_view else "自由相机"
            print(f">>> Viewer 已切换: {mode}")
            return
        if key in (glfw.KEY_R, glfw.KEY_BACKSPACE):
            pending_actions.append(("reset", "", 0.0))
            return
        if key == glfw.KEY_1:
            pending_actions.append(("switch_mode", PICKCUP_MODE, 0.0))
            return
        if key == glfw.KEY_2:
            pending_actions.append(("switch_mode", LIFTBAG_MODE, 0.0))
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
        if key in pc.KEY_BINDINGS:
            joint_name, delta = pc.KEY_BINDINGS[key]
            pending_actions.append(("joint", joint_name, delta))

    def process_pending_actions() -> None:
        nonlocal current_mode, next_record_time, hold_accum, success_announced

        while pending_actions:
            action, joint_name, delta = pending_actions.popleft()

            if action == "switch_mode":
                if joint_name == current_mode:
                    label = "[1] 拿杯子" if current_mode == PICKCUP_MODE else "[2] 拎包"
                    print(f">>> 已经处于 {label} 模式")
                    continue
                recorders[current_mode].close_session()
                current_mode = joint_name
                reset_scene(
                    model,
                    data,
                    robot_ctrl,
                    actuator_indices,
                    scene=scene,
                    robot_position=robot_position_for_mode(current_mode),
                )
                cup_grasp.reset()
                bag_grasp.reset()
                next_record_time = 0.0
                hold_accum = 0.0
                success_announced = False
                session_root = recorders[current_mode].start_session()
                label = "[1] 拿杯子" if current_mode == PICKCUP_MODE else "[2] 拎包"
                print(f">>> 已切换到 {label} 模式")
                print(f">>> 新 LeRobot session: {session_root}")
                continue

            if action == "reset":
                recorders[current_mode].restart_session()
                reset_scene(
                    model,
                    data,
                    robot_ctrl,
                    actuator_indices,
                    scene=scene,
                    robot_position=robot_position_for_mode(current_mode),
                )
                cup_grasp.reset()
                bag_grasp.reset()
                next_record_time = 0.0
                hold_accum = 0.0
                success_announced = False
                print(">>> 场景已重置，已开启新 LeRobot session")
                continue

            if action == "gripper_open":
                ok = pc.try_update_teleop_targets(
                    model, data, robot_ctrl, pc.GRIPPER_OPEN, teleop_geoms, obstacle_geoms
                )
                print(">>> 任务夹爪已张开" if ok else ">>> 任务夹爪无法继续张开（碰撞限制）")
                continue
            if action == "gripper_close":
                ok = pc.try_update_teleop_targets(
                    model, data, robot_ctrl, pc.GRIPPER_CLOSED, teleop_geoms, obstacle_geoms
                )
                print(">>> 任务夹爪已闭合" if ok else ">>> 任务夹爪无法继续闭合（碰撞限制）")
                continue

            reach_map = {
                "reach_forward": ("右臂前伸", "无法继续前伸", pc.REACH_FORWARD_DELTAS),
                "reach_backward": ("右臂后收", "无法继续后收", pc.REACH_BACKWARD_DELTAS),
                "reach_up": ("右臂抬起", "无法继续抬起", pc.REACH_UP_DELTAS),
                "reach_down": ("右臂下压", "无法继续下压", pc.REACH_DOWN_DELTAS),
            }
            if action in reach_map:
                ok_msg, fail_msg, deltas = reach_map[action]
                updates = {j: robot_ctrl[j] + d for j, d in deltas.items()}
                ok = pc.try_update_teleop_targets(
                    model, data, robot_ctrl, updates, teleop_geoms, obstacle_geoms
                )
                print(f">>> {ok_msg}" if ok else f">>> 右臂已达碰撞极限，{fail_msg}")
                continue

            if action == "joint":
                proposed = robot_ctrl[joint_name] + delta
                ok = pc.try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    {joint_name: proposed},
                    teleop_geoms,
                    obstacle_geoms,
                )
                if ok:
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
        pc.set_viewer_robot_camera(viewer, model, use_robot_camera_view)
        preview_inited = False
        try:
            while viewer.is_running() and not should_exit:
                loop_start = time.time()
                preview_rgb: np.ndarray | None = None

                with viewer.lock():
                    process_pending_actions()
                    pc.apply_hold_ctrl(model, data, robot_ctrl, actuator_indices)
                    pc.apply_kinematic_teleop(model, data, robot_ctrl)

                    if not is_paused:
                        mujoco.mj_step(model, data)
                        pc.apply_kinematic_teleop(model, data, robot_ctrl)
                        if current_mode == PICKCUP_MODE:
                            pc.sanitize_cup_physics(model, data, grasp_active=cup_grasp.active)
                            sanitize_bag_physics(model, data, grasp_active=False)
                            pc.update_grasp(
                                model,
                                data,
                                robot_ctrl,
                                cup_grasp,
                                gripper_geoms,
                                cup_geoms,
                            )
                        else:
                            pc.sanitize_cup_physics(model, data, grasp_active=False)
                            sanitize_bag_physics(model, data, grasp_active=bag_grasp.active)
                            update_bag_grasp(
                                model,
                                data,
                                robot_ctrl,
                                bag_grasp,
                                gripper_geoms,
                                handle_geoms,
                            )
                        mujoco.mj_forward(model, data)

                        if (
                            current_mode == LIFTBAG_MODE
                            and bag_grasp.active
                            and bag_lift_height(model, data) >= LIFT_HEIGHT_MIN
                        ):
                            hold_accum += model.opt.timestep
                            if hold_accum >= HOLD_TIME_S and not success_announced:
                                success_announced = True
                                print(
                                    f">>> 任务成功：包已抬起 ≥{LIFT_HEIGHT_MIN * 100:.0f}cm "
                                    f"并保持 {HOLD_TIME_S:.1f}s"
                                )
                        else:
                            hold_accum = 0.0

                        next_record_time, preview_rgb = maybe_record_frame(
                            model,
                            data,
                            robot_ctrl,
                            renderer,
                            recorders[current_mode],
                            next_record_time,
                            PICKCUP_TASK if current_mode == PICKCUP_MODE else LIFTBAG_TASK,
                        )

                    pc._configure_viewer(viewer)
                    pc.set_viewer_robot_camera(viewer, model, use_robot_camera_view)

                viewer.sync()

                if not preview_inited:
                    pc.init_head_camera_preview(enable_by_default=False)
                    preview_inited = True

                if pc._head_camera_preview_enabled:
                    now = time.time()
                    if preview_rgb is not None:
                        pc.show_head_camera_preview(preview_rgb)
                        next_preview_wall = now + pc.RECORD_PERIOD
                    elif now >= next_preview_wall:
                        pc.show_head_camera_preview(pc.render_head_camera(renderer, data))
                        next_preview_wall = now + pc.RECORD_PERIOD

                with viewer.lock():
                    pc._configure_viewer(viewer)
                    pc.set_viewer_robot_camera(viewer, model, use_robot_camera_view)

                sleep_time = model.opt.timestep - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n\n用户中断")
        finally:
            print("\n\n正在保存 LeRobot dataset...")
            for recorder in recorders.values():
                recorder.close_session()
            pc.close_head_camera_preview()

    print("\n程序退出")


if __name__ == "__main__":
    main()
