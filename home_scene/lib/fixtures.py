"""程序化夹具 / 小物体。id 稳定，日后可点状换成 RoboCasa mesh。"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Sequence

from cup_object import CupSpec, add_colored_cup, _contact_attrs
from g1_mount import ensure_child, fmt, yaw_to_quat


def add_box(
    parent: ET.Element,
    name: str,
    pos: Sequence[float],
    size: Sequence[float],
    material: str,
    *,
    collide: bool = True,
) -> ET.Element:
    return ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": fmt(pos),
            "size": fmt(size),
            "material": material,
            "contype": "1" if collide else "0",
            "conaffinity": "1" if collide else "0",
        },
    )


def add_scene_materials(root: ET.Element) -> None:
    asset = ensure_child(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.82 0.86 0.90",
            "rgb2": "0.30 0.34 0.38",
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
            "builtin": "checker",
            "rgb1": "0.70 0.68 0.64",
            "rgb2": "0.62 0.60 0.56",
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
            "texrepeat": "14 8",
            "reflectance": "0.02",
        },
    )
    existing = {m.get("name") for m in asset.findall("material")}
    for name, rgba in {
        "wall_paint": "0.92 0.92 0.90 1",
        "counter_top": "0.55 0.55 0.52 1",
        "cabinet_body": "0.78 0.62 0.42 1",
        "cabinet_door": "0.70 0.55 0.38 1",
        "sink_steel": "0.72 0.74 0.76 1",
        "faucet_chrome": "0.85 0.86 0.88 1",
        "stove_black": "0.12 0.12 0.13 1",
        "stove_grate": "0.25 0.25 0.26 1",
        "burner_off": "0.15 0.15 0.15 1",
        "knob_plastic": "0.9 0.9 0.92 1",
        "backsplash": "0.88 0.90 0.92 1",
        "window_glass": "0.55 0.72 0.88 0.35",
        "window_frame": "0.45 0.40 0.35 1",
        "bowl_ceramic": "0.94 0.94 0.90 1",
        "bottle_green": "0.20 0.45 0.28 1",
        "switch_plate": "0.95 0.95 0.93 1",
        "switch_rocker": "0.85 0.85 0.82 1",
        "lamp_shade": "0.96 0.94 0.88 1",
        "table_wood": "0.62 0.42 0.28 1",
        "sofa_fabric": "0.42 0.48 0.52 1",
        "door_frame": "0.50 0.42 0.34 1",
    }.items():
        if name not in existing:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


# 旧名兼容
add_kitchen_prop_materials = add_scene_materials


def add_counter(
    worldbody: ET.Element,
    *,
    center_xy: tuple[float, float],
    length: float,
    depth: float,
    top_z: float,
) -> None:
    cx, cy = center_xy
    body_half_z = top_z / 2.0 - 0.02
    add_box(
        worldbody,
        "counter_body",
        (cx, cy, body_half_z),
        (length / 2.0, depth / 2.0, body_half_z),
        "cabinet_body",
    )
    add_box(
        worldbody,
        "counter_top",
        (cx, cy, top_z - 0.015),
        (length / 2.0 + 0.01, depth / 2.0 + 0.01, 0.015),
        "counter_top",
    )


def add_cabinet_stub(
    worldbody: ET.Element,
    *,
    origin_xy: tuple[float, float],
    depth: float,
    width: float,
    height: float,
) -> None:
    cab_x, cab_y = origin_xy
    cab = ET.SubElement(
        worldbody,
        "body",
        {"name": "cabinet_stub", "pos": fmt((cab_x, cab_y, 0.0))},
    )
    add_box(
        cab,
        "cabinet_stub_body",
        (0.0, 0.0, height / 2.0),
        (depth / 2.0, width / 2.0, height / 2.0),
        "cabinet_body",
    )
    door = ET.SubElement(
        cab,
        "body",
        {
            "name": "cabinet_stub_door",
            "pos": fmt((depth / 2.0, 0.0, height / 2.0)),
        },
    )
    ET.SubElement(
        door,
        "joint",
        {
            "name": "cabinet_stub_hinge",
            "type": "hinge",
            "axis": "0 0 1",
            "range": f"0 {math.pi * 0.6:.4f}",
            "damping": "0.5",
        },
    )
    add_box(
        door,
        "cabinet_stub_door_geom",
        (0.012, 0.0, 0.0),
        (0.012, width / 2.0 - 0.01, height / 2.0 - 0.02),
        "cabinet_door",
    )


def add_backsplash_and_window(
    worldbody: ET.Element, *, counter_y: float, half_x: float, wall_y: float
) -> None:
    """wall_y 为北墙世界 Y（正值）；窗开在南墙。"""
    del counter_y
    add_box(
        worldbody,
        "backsplash",
        (0.0, wall_y - 0.02, 1.15),
        (half_x - 0.15, 0.015, 0.28),
        "backsplash",
        collide=False,
    )
    add_box(
        worldbody,
        "window_frame",
        (0.8, -wall_y + 0.03, 1.45),
        (0.55, 0.03, 0.45),
        "window_frame",
        collide=False,
    )
    add_box(
        worldbody,
        "window_glass",
        (0.8, -wall_y + 0.055, 1.45),
        (0.48, 0.01, 0.38),
        "window_glass",
        collide=False,
    )


def add_sink_station(
    worldbody: ET.Element, *, cx: float, cy: float, top_z: float
) -> None:
    add_box(
        worldbody,
        "sink_basin",
        (cx, cy, top_z - 0.08),
        (0.28, 0.20, 0.07),
        "sink_steel",
    )
    add_box(
        worldbody,
        "sink_rim",
        (cx, cy, top_z - 0.005),
        (0.30, 0.22, 0.008),
        "sink_steel",
    )
    faucet = ET.SubElement(
        worldbody,
        "body",
        {"name": "faucet", "pos": fmt((cx, cy + 0.18, top_z))},
    )
    ET.SubElement(
        faucet,
        "geom",
        {
            "name": "faucet_base",
            "type": "cylinder",
            "size": "0.025 0.04",
            "pos": "0 0 0.04",
            "material": "faucet_chrome",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        faucet,
        "geom",
        {
            "name": "faucet_neck",
            "type": "capsule",
            "fromto": "0 0 0.08  0 -0.12 0.22",
            "size": "0.015",
            "material": "faucet_chrome",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def add_stove_station(
    worldbody: ET.Element, *, cx: float, cy: float, top_z: float
) -> None:
    add_box(
        worldbody,
        "stove_body",
        (cx, cy, top_z - 0.02),
        (0.38, 0.28, 0.03),
        "stove_black",
    )
    burner_offsets = [(-0.14, -0.08), (0.14, -0.08), (-0.14, 0.08), (0.14, 0.08)]
    for i, (bx, by) in enumerate(burner_offsets):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"stove_burner_{i}",
                "type": "cylinder",
                "pos": fmt((cx + bx, cy + by, top_z + 0.012)),
                "size": "0.07 0.008",
                "material": "burner_off",
                "contype": "1",
                "conaffinity": "1",
            },
        )
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"stove_grate_{i}",
                "type": "cylinder",
                "pos": fmt((cx + bx, cy + by, top_z + 0.022)),
                "size": "0.085 0.004",
                "material": "stove_grate",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    for i, kx in enumerate((-0.22, -0.08, 0.08, 0.22)):
        knob = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"stove_knob_{i}",
                "pos": fmt((cx + kx, cy - 0.30, top_z + 0.02)),
            },
        )
        ET.SubElement(
            knob,
            "joint",
            {
                "name": f"stove_knob_{i}_hinge",
                "type": "hinge",
                "axis": "0 1 0",
                "range": f"0 {math.pi * 0.75:.4f}",
                "damping": "0.2",
            },
        )
        ET.SubElement(
            knob,
            "geom",
            {
                "name": f"stove_knob_{i}_geom",
                "type": "cylinder",
                "size": "0.018 0.015",
                "material": "knob_plastic",
                # 旋钮用按键切换，不做物理碰撞，避免手臂扫到时 QACC 爆炸
                "contype": "0",
                "conaffinity": "0",
            },
        )


def add_ceiling_lamp_and_switch(
    worldbody: ET.Element, *, switch_pos: Sequence[float]
) -> None:
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "ceiling_lamp",
            "pos": "0 0.3 2.35",
            "dir": "0 0 -1",
            "diffuse": "0.95 0.92 0.85",
            "specular": "0.2 0.2 0.2",
            "cutoff": "70",
            "castshadow": "true",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "lamp_shade",
            "type": "cylinder",
            "pos": "0 0.3 2.28",
            "size": "0.18 0.06",
            "material": "lamp_shade",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    plate = ET.SubElement(
        worldbody,
        "body",
        {"name": "light_switch", "pos": fmt(switch_pos)},
    )
    add_box(plate, "light_switch_plate", (0.0, 0.0, 0.0), (0.04, 0.01, 0.06), "switch_plate")
    rocker = ET.SubElement(
        plate,
        "body",
        {"name": "light_switch_rocker", "pos": "0 0.012 0"},
    )
    ET.SubElement(
        rocker,
        "joint",
        {
            "name": "light_switch_slide",
            "type": "slide",
            "axis": "0 0 1",
            "range": "0 0.02",
            "damping": "0.5",
        },
    )
    add_box(
        rocker,
        "light_switch_rocker_geom",
        (0.0, 0.0, 0.0),
        (0.018, 0.008, 0.022),
        "switch_rocker",
    )


def add_coffee_table(
    worldbody: ET.Element,
    *,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float] = (1.10, 0.60),
    top_z: float = 0.42,
) -> None:
    cx, cy = center_xy
    half_x, half_y = size_xy[0] / 2.0, size_xy[1] / 2.0
    add_box(
        worldbody,
        "coffee_table_top",
        (cx, cy, top_z - 0.02),
        (half_x, half_y, 0.02),
        "table_wood",
    )
    for i, (lx, ly) in enumerate(
        ((-half_x + 0.06, -half_y + 0.06), (half_x - 0.06, -half_y + 0.06),
         (-half_x + 0.06, half_y - 0.06), (half_x - 0.06, half_y - 0.06))
    ):
        add_box(
            worldbody,
            f"coffee_table_leg_{i}",
            (cx + lx, cy + ly, (top_z - 0.04) / 2.0),
            (0.03, 0.03, (top_z - 0.04) / 2.0),
            "table_wood",
        )


def add_sofa_stub(
    worldbody: ET.Element,
    *,
    center_xy: tuple[float, float],
    yaw: float = 0.0,
) -> None:
    """客厅靠东墙的矮沙发占位（视觉+碰撞，无关节）。"""
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "sofa_stub",
            "pos": fmt((center_xy[0], center_xy[1], 0.0)),
            "quat": fmt(yaw_to_quat(yaw)),
        },
    )
    add_box(body, "sofa_seat", (0.0, 0.0, 0.20), (0.38, 0.85, 0.20), "sofa_fabric")
    add_box(body, "sofa_back", (0.30, 0.0, 0.46), (0.08, 0.85, 0.26), "sofa_fabric")


def add_floor_lamp(
    worldbody: ET.Element, *, pos_xy: tuple[float, float]
) -> None:
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "living_lamp",
            "pos": fmt((pos_xy[0], pos_xy[1], 1.70)),
            "dir": "0 0 -1",
            "diffuse": "0.55 0.52 0.45",
            "cutoff": "60",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "living_lamp_pole",
            "type": "cylinder",
            "pos": fmt((pos_xy[0], pos_xy[1], 0.75)),
            "size": "0.02 0.75",
            "material": "faucet_chrome",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "living_lamp_shade",
            "type": "cylinder",
            "pos": fmt((pos_xy[0], pos_xy[1], 1.58)),
            "size": "0.12 0.08",
            "material": "lamp_shade",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def add_bowl(
    worldbody: ET.Element,
    *,
    name: str,
    position: Sequence[float],
    yaw: float = 0.0,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": name, "pos": fmt(position), "quat": fmt(yaw_to_quat(yaw))},
    )
    ET.SubElement(
        body,
        "joint",
        {"name": f"{name}_free", "type": "free", "damping": "0.2", "armature": "0.001"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_geom",
            "type": "cylinder",
            "size": "0.07 0.035",
            "material": "bowl_ceramic",
            "mass": "0.25",
            **_contact_attrs(),
        },
    )


def add_bottle(
    worldbody: ET.Element,
    *,
    name: str,
    position: Sequence[float],
    yaw: float = 0.0,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": name, "pos": fmt(position), "quat": fmt(yaw_to_quat(yaw))},
    )
    ET.SubElement(
        body,
        "joint",
        {"name": f"{name}_free", "type": "free", "damping": "0.15", "armature": "0.001"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_body",
            "type": "cylinder",
            "size": "0.035 0.10",
            "pos": "0 0 0.10",
            "material": "bottle_green",
            "mass": "0.35",
            **_contact_attrs(),
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_neck",
            "type": "cylinder",
            "size": "0.015 0.03",
            "pos": "0 0 0.23",
            "material": "bottle_green",
            "mass": "0.05",
            **_contact_attrs(),
        },
    )


def default_kitchen_small_objects(counter_y: float, top_z: float) -> list[tuple[str, object]]:
    return [
        ("cup", CupSpec("red", offset_xy=(-0.15, -0.05))),
        ("cup", CupSpec("blue", offset_xy=(0.05, 0.02))),
        ("cup", CupSpec("white", offset_xy=(0.25, -0.08))),
        ("bowl", {"name": "bowl_0", "position": (0.45, counter_y - 0.05, top_z + 0.035)}),
        ("bottle", {"name": "bottle_0", "position": (-0.45, counter_y + 0.05, top_z + 0.002)}),
    ]


def place_small_objects(
    worldbody: ET.Element,
    items: Sequence[tuple[str, object]],
    *,
    counter_y: float,
    top_z: float,
) -> None:
    cup_z = top_z + 0.002
    for kind, spec in items:
        if kind == "cup":
            assert isinstance(spec, CupSpec)
            ox, oy = spec.offset_xy
            add_colored_cup(
                worldbody,
                spec,
                position=(ox, counter_y + oy, cup_z),
                body_name=spec.name or f"cup_{spec.color}",
            )
        elif kind == "bowl":
            add_bowl(worldbody, **spec)  # type: ignore[arg-type]
        elif kind == "bottle":
            add_bottle(worldbody, **spec)  # type: ignore[arg-type]
        else:
            raise ValueError(f"未知小物体 kind: {kind}")
