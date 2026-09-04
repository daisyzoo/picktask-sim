"""彩色杯子物体（程序化圆柱叠层，自包含，不依赖 picktask）。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Sequence

from g1_mount import ensure_child, fmt, yaw_to_quat

CUP_COLORS: dict[str, str] = {
    "red": "0.86 0.12 0.15 1",
    "blue": "0.18 0.42 0.86 1",
    "green": "0.18 0.62 0.32 1",
    "white": "0.94 0.94 0.92 1",
    "yellow": "0.92 0.78 0.12 1",
    "orange": "0.92 0.45 0.12 1",
}

CUP_INNER_RGBA = "0.98 0.98 0.96 1"
CONTACT_FRICTION = "1.2 0.08 0.002"

# (suffix, radius, half_height, use_outer_color, mass, z)
_CUP_SEGMENTS: tuple[tuple[str, float, float, bool, float, float], ...] = (
    ("bottom", 0.030, 0.005, False, 0.006, 0.0025),
    ("lower", 0.032, 0.014, True, 0.008, 0.020),
    ("mid1", 0.034, 0.014, True, 0.008, 0.045),
    ("mid2", 0.036, 0.014, True, 0.008, 0.070),
    ("rim", 0.040, 0.006, False, 0.004, 0.096),
)


@dataclass(frozen=True)
class CupSpec:
    color: str
    # 相对台面中心的水平偏移；z 由台面高度决定
    offset_xy: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.color not in CUP_COLORS:
            raise ValueError(f"未知杯子颜色: {self.color}; 可选 {sorted(CUP_COLORS)}")


def material_name(color: str) -> str:
    return f"cup_{color}"


def add_cup_materials(root: ET.Element) -> None:
    asset = ensure_child(root, "asset")
    existing = {m.get("name") for m in asset.findall("material")}
    if "cup_inner" not in existing:
        ET.SubElement(asset, "material", {"name": "cup_inner", "rgba": CUP_INNER_RGBA})
    for color, rgba in CUP_COLORS.items():
        name = material_name(color)
        if name not in existing:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _contact_attrs() -> dict[str, str]:
    return {
        "contype": "1",
        "conaffinity": "1",
        "friction": CONTACT_FRICTION,
        "condim": "4",
        "solref": "0.02 1",
        "solimp": "0.9 0.95 0.001",
    }


def add_colored_cup(
    worldbody: ET.Element,
    spec: CupSpec,
    *,
    position: Sequence[float],
    body_name: str | None = None,
) -> ET.Element:
    """在 worldbody 添加带 free joint 的彩色杯子。"""
    name = body_name or spec.name or f"cup_{spec.color}"
    outer = material_name(spec.color)
    cup = ET.SubElement(
        worldbody,
        "body",
        {
            "name": name,
            "pos": fmt(position),
            "quat": fmt(yaw_to_quat(spec.yaw)),
        },
    )
    ET.SubElement(
        cup,
        "joint",
        {
            "name": f"{name}_free",
            "type": "free",
            "damping": "0.15",
            "armature": "0.001",
        },
    )
    for suffix, radius, half_h, use_outer, mass, z in _CUP_SEGMENTS:
        mat = outer if use_outer else "cup_inner"
        ET.SubElement(
            cup,
            "geom",
            {
                "name": f"{name}_{suffix}",
                "type": "cylinder",
                "size": fmt((radius, half_h)),
                "material": mat,
                "mass": f"{mass:.6g}",
                "pos": f"0 0 {z:.6g}",
                **_contact_attrs(),
            },
        )
    return cup


def default_counter_cups() -> tuple[CupSpec, ...]:
    """北侧台面上默认摆放的多色杯子。"""
    return (
        CupSpec("red", offset_xy=(-0.35, -0.05)),
        CupSpec("blue", offset_xy=(-0.10, 0.02)),
        CupSpec("green", offset_xy=(0.15, -0.08)),
        CupSpec("yellow", offset_xy=(0.40, 0.00)),
    )
