#!/usr/bin/env python3
"""编译公寓 stub MJCF，校验工位与房间；不打开 viewer。"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from asset_registry import default_registry  # noqa: E402
from g1_mount import ACTUATED_JOINTS, sync_g1_ctrl_to_qpos  # noqa: E402
from layout import APARTMENT, LAYOUT_CHOICES, layout_from_name  # noqa: E402
from paths import CACHE_DIR, G1_URDF_PATH, HOME_SCENE_DIR, assert_decoupled  # noqa: E402
from scene_composer import (  # noqa: E402
    COUNTER_TOP_Z,
    ROOM_SIZE_X,
    ROOM_SIZE_Y,
    SceneLayout,
    _counter_front_y,
    build_xml,
    make_model_data,
)


def _assert_room_not_under_pelvis(xml: str) -> None:
    root = ET.fromstring(xml)
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError("缺少 pelvis")
    under = {g.get("name") for g in pelvis.findall("./geom")}
    under |= {b.get("name") for b in pelvis.findall("./body")}
    banned = {
        "floor",
        "wall_north",
        "counter_body",
        "counter_top",
        "cabinet_stub",
        "coffee_table_top",
        "sofa_stub",
    }
    hit = under & banned
    if hit:
        raise RuntimeError(f"场景节点被错误挂到 pelvis 下: {sorted(hit)}")
    world = root.find("worldbody")
    assert world is not None
    world_geoms = {g.get("name") for g in world.findall("geom")}
    if "floor" not in world_geoms:
        raise RuntimeError("floor 必须是 worldbody 的直接子 geom")
    if "coffee_table_top" not in world_geoms:
        raise RuntimeError("客厅茶几缺失")
    if "counter_top" not in world_geoms:
        raise RuntimeError("厨房台面缺失")


def _assert_workstations() -> None:
    kitchen = APARTMENT.room("kitchen")
    living = APARTMENT.room("living")
    names = {ws.name for ws in kitchen.workstations} | {ws.name for ws in living.workstations}
    required = {"counter_front", "cabinet_west", "table_front", "sink_front", "stove_front"}
    missing = required - names
    if missing:
        raise RuntimeError(f"工位缺失: {sorted(missing)}")
    table = living.workstation("table_front")
    if table.position[0] <= 3.0:
        raise RuntimeError(f"客厅工位应在门洞东侧，实际 x={table.position[0]:.2f}")
    door = APARTMENT.portals[0]
    if door.from_room != "kitchen" or door.to_room != "living":
        raise RuntimeError("门洞应连接 kitchen → living")


def _compile(name: str) -> None:
    layout = layout_from_name(name)
    xml = build_xml(layout)
    _assert_room_not_under_pelvis(xml)
    model, data = make_model_data(layout)
    import mujoco

    g1_acts = sum(
        1
        for jn in ACTUATED_JOINTS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_pos") >= 0
    )
    if g1_acts != 23:
        raise RuntimeError(f"{name}: 期望 23 个 G1 执行器，实际 {g1_acts}")
    sync_g1_ctrl_to_qpos(model, data)
    for _ in range(10):
        sync_g1_ctrl_to_qpos(model, data)
        mujoco.mj_step(model, data)
    print(
        f"  layout={name:16s} ws={layout.workstation:14s} "
        f"pos=({layout.robot_position[0]:+.2f},{layout.robot_position[1]:+.2f}) "
        f"nq={model.nq} nbody={model.nbody} ngeom={model.ngeom}"
    )


def main() -> int:
    assert_decoupled()
    print(f"home_scene: {HOME_SCENE_DIR}")
    print(f"G1 URDF:    {G1_URDF_PATH}  exists={G1_URDF_PATH.is_file()}")

    reg = default_registry()
    print(f"home:       {APARTMENT.home_id}  rooms={[r.room_id for r in APARTMENT.rooms]}")
    print(
        f"kitchen:    {ROOM_SIZE_X:.1f}m x {ROOM_SIZE_Y:.1f}m  "
        f"counter_top_z={COUNTER_TOP_Z:.2f}m  front_y={_counter_front_y():.2f}"
    )
    kitchen_ws = APARTMENT.room("kitchen").workstation("counter_front")
    print(
        f"robot:      counter_front=({kitchen_ws.position[0]:.2f},{kitchen_ws.position[1]:.2f})  "
        f"clearance≈{_counter_front_y() - kitchen_ws.position[1]:.2f}m"
    )
    _assert_workstations()
    print(f"layouts:    {', '.join(LAYOUT_CHOICES)}")

    layout = SceneLayout.kitchen_stub()
    xml = build_xml(layout, reg)
    _assert_room_not_under_pelvis(xml)
    out = CACHE_DIR / "kitchen_stub_latest.xml"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    print(f"wrote XML:  {out}  ({len(xml)} chars)")

    print("compile:")
    for name in LAYOUT_CHOICES:
        _compile(name)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
