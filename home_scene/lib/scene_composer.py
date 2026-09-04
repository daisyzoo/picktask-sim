"""场景拼装编排：Layout → HomeSpec → MJCF → MjModel。

几何细节在 fixtures / room_builder；此处只挂 G1、铺房、写 viewer 选项。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco

from asset_registry import AssetRegistry, default_registry
from cup_object import add_cup_materials
from fixtures import add_scene_materials
from g1_mount import ensure_child, fmt, mount_g1
from layout import (
    APARTMENT,
    COUNTER_TOP_Z,
    LAYOUT_CHOICES,
    ROOM_SIZE_X,
    ROOM_SIZE_Y,
    SceneLayout,
    cabinet_origin,
    close_cabinet_robot_pose,
    counter_front_y,
    default_robot_pose,
    layout_from_name,
)
from robocasa_bridge import status as robocasa_status
from room_builder import add_home_world, apartment_extent


def _counter_front_y() -> float:
    return counter_front_y(APARTMENT.room("kitchen"))


def _add_world_options(root: ET.Element, layout: SceneLayout) -> None:
    home = layout.resolve()[0]
    center, extent = apartment_extent(home)

    option = ensure_child(root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "RK4")

    statistic = ensure_child(root, "statistic")
    statistic.set("center", fmt(center))
    statistic.set("extent", f"{extent:.2f}")

    visual = ensure_child(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.35 0.35 0.35",
            "diffuse": "0.7 0.7 0.7",
            "specular": "0.15 0.15 0.15",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.12 0.14 0.16 1"})
    ET.SubElement(
        visual,
        "global",
        {"azimuth": "-145", "elevation": "-20", "offwidth": "960", "offheight": "720"},
    )
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    ET.SubElement(visual, "map", {"znear": "0.01", "zfar": "80"})


def build_xml(
    layout: SceneLayout | None = None,
    registry: AssetRegistry | None = None,
) -> str:
    """生成完整公寓 stub MJCF。

    顺序：先挂 G1（仅机器人进 pelvis），再铺房间，避免家具被包进 pelvis。
    """
    layout = layout or SceneLayout.kitchen_stub()
    registry = registry or default_registry()
    home, _room, _ws = layout.resolve()
    _ = registry.get(home.home_id)
    _ = robocasa_status()

    root = ET.Element("mujoco", {"model": f"home_scene_{home.home_id.replace('.', '_')}"})
    _add_world_options(root, layout)
    add_scene_materials(root)
    add_cup_materials(root)
    mount_g1(
        root,
        robot_position=layout.robot_position,
        robot_yaw=layout.robot_yaw,
    )
    add_home_world(root, home, layout)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def make_model_data(
    layout: SceneLayout | None = None,
    registry: AssetRegistry | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    layout = layout or SceneLayout.kitchen_stub()
    model = mujoco.MjModel.from_xml_string(build_xml(layout, registry))
    data = mujoco.MjData(model)
    if layout.cabinet_open_angle > 0:
        from tasks.close_cabinet import CloseCabinetTask

        task = CloseCabinetTask(open_angle=layout.cabinet_open_angle)
        task.reset(model, data)
    from appliances import set_light_switch, sync_appliances

    set_light_switch(model, data, on=True)
    sync_appliances(model, data)
    return model, data


def configure_viewer(viewer) -> None:
    """关闭调试可视化；passive viewer 需在主循环内反复调用。"""
    disable_flags = (
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_JOINT,
        mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        mujoco.mjtVisFlag.mjVIS_COM,
        mujoco.mjtVisFlag.mjVIS_CAMERA,
        mujoco.mjtVisFlag.mjVIS_LIGHT,
    )
    for flag in disable_flags:
        viewer.opt.flags[flag] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE


__all__ = [
    "APARTMENT",
    "COUNTER_TOP_Z",
    "LAYOUT_CHOICES",
    "ROOM_SIZE_X",
    "ROOM_SIZE_Y",
    "SceneLayout",
    "build_xml",
    "cabinet_origin",
    "close_cabinet_robot_pose",
    "configure_viewer",
    "default_robot_pose",
    "layout_from_name",
    "make_model_data",
    "_counter_front_y",
]
