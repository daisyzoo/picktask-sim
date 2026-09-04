"""按 HomeSpec 铺房间外壳与夹具。scene_composer 只负责编排。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from fixtures import (
    add_backsplash_and_window,
    add_box,
    add_cabinet_stub,
    add_ceiling_lamp_and_switch,
    add_coffee_table,
    add_counter,
    add_floor_lamp,
    add_sink_station,
    add_sofa_stub,
    add_stove_station,
    default_kitchen_small_objects,
    place_small_objects,
)
from g1_mount import ensure_child, fmt
from layout import (
    CABINET_DEPTH,
    CABINET_HEIGHT,
    CABINET_WIDTH,
    COUNTER_DEPTH,
    COUNTER_LENGTH,
    COUNTER_TOP_Z,
    SINK_X_LOCAL,
    STOVE_X_LOCAL,
    WALL_HEIGHT,
    WALL_THICKNESS,
    HomeSpec,
    Portal,
    RoomSpec,
    SceneLayout,
    cabinet_origin,
    counter_center_y,
)


def _wall_center_and_size(
    room: RoomSpec, side: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ox, oy = room.origin_xy
    half_x, half_y = room.half()
    h = room.wall_height / 2.0
    t = WALL_THICKNESS / 2.0
    if side == "north":
        return (ox, oy + half_y, h), (half_x, t, h)
    if side == "south":
        return (ox, oy - half_y, h), (half_x, t, h)
    if side == "east":
        return (ox + half_x, oy, h), (t, half_y, h)
    if side == "west":
        return (ox - half_x, oy, h), (t, half_y, h)
    raise ValueError(f"未知墙面: {side}")


def _wall_geom_name(room: RoomSpec, side: str, suffix: str = "") -> str:
    if room.room_id == "kitchen" and not suffix:
        return f"wall_{side}"
    if suffix:
        return f"wall_{room.room_id}_{side}_{suffix}"
    return f"wall_{room.room_id}_{side}"


def _add_full_wall(worldbody: ET.Element, room: RoomSpec, side: str) -> None:
    pos, size = _wall_center_and_size(room, side)
    add_box(worldbody, _wall_geom_name(room, side), pos, size, "wall_paint")


def _add_split_wall_with_door(
    worldbody: ET.Element, room: RoomSpec, portal: Portal
) -> None:
    """把 from_room 的一面墙拆成门洞两侧 + 门楣。"""
    ox, oy = room.origin_xy
    half_x, half_y = room.half()
    h = room.wall_height / 2.0
    t = WALL_THICKNESS / 2.0
    door_half = portal.width / 2.0
    along = portal.along

    if portal.wall == "east":
        wall_x = ox + half_x
        wall_span = half_y
        center_along = oy + along
        lo, hi = center_along - door_half, center_along + door_half
        # 沿 Y：房间 [oy-half_y, oy+half_y]
        segs = []
        south_hi = lo
        north_lo = hi
        room_lo, room_hi = oy - wall_span, oy + wall_span
        if south_hi > room_lo + 1e-4:
            mid = 0.5 * (room_lo + south_hi)
            segs.append((mid, 0.5 * (south_hi - room_lo), "s"))
        if room_hi > north_lo + 1e-4:
            mid = 0.5 * (north_lo + room_hi)
            segs.append((mid, 0.5 * (room_hi - north_lo), "n"))
        for mid_y, half_len, tag in segs:
            add_box(
                worldbody,
                _wall_geom_name(room, "east", tag),
                (wall_x, mid_y, h),
                (t, half_len, h),
                "wall_paint",
            )
        lintel_half_z = (room.wall_height - portal.height) / 2.0
        add_box(
            worldbody,
            f"{portal.name}_lintel",
            (wall_x, center_along, portal.height + lintel_half_z),
            (t, door_half, lintel_half_z),
            "door_frame",
        )
        for tag, y in (("south", lo), ("north", hi)):
            add_box(
                worldbody,
                f"{portal.name}_jamb_{tag}",
                (wall_x, y, portal.height / 2.0),
                (t + 0.01, 0.03, portal.height / 2.0),
                "door_frame",
            )
        return
    raise NotImplementedError(f"暂只支持东墙门洞，收到 {portal.wall}")


def _add_floor(worldbody: ET.Element, home: HomeSpec) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for room in home.rooms:
        hx, hy = room.half()
        xs.extend((room.origin_xy[0] - hx, room.origin_xy[0] + hx))
        ys.extend((room.origin_xy[1] - hy, room.origin_xy[1] + hy))
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": fmt((cx, cy, 0.0)),
            "size": fmt(((max_x - min_x) / 2.0 + 0.8, (max_y - min_y) / 2.0 + 0.8, 0.05)),
            "material": "floor_material",
            "contype": "1",
            "conaffinity": "1",
        },
    )


def _add_env_lights(worldbody: ET.Element) -> None:
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": fmt((-2.0, -2.5, 4.0)),
            "dir": "0.35 0.45 -1",
            "directional": "true",
            "diffuse": "0.35 0.35 0.34",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill_light",
            "pos": fmt((4.5, 2.0, 3.0)),
            "dir": "-0.45 -0.25 -1",
            "directional": "true",
            "diffuse": "0.12 0.12 0.14",
        },
    )


def _portal_walls(home: HomeSpec, room: RoomSpec) -> dict[str, Portal]:
    return {
        p.wall: p
        for p in home.portals
        if p.from_room == room.room_id
    }


def _add_room_shell(worldbody: ET.Element, home: HomeSpec, room: RoomSpec) -> None:
    punched = _portal_walls(home, room)
    skip_west = any(
        p.to_room == room.room_id and p.wall == "east" for p in home.portals
    )
    for side in ("north", "south", "east", "west"):
        if side == "west" and skip_west:
            continue
        portal = punched.get(side)
        if portal is not None:
            _add_split_wall_with_door(worldbody, room, portal)
        else:
            _add_full_wall(worldbody, room, side)


def add_kitchen_interior(worldbody: ET.Element, room: RoomSpec, layout: SceneLayout) -> None:
    cy = counter_center_y(room)
    ox, _ = room.origin_xy
    half_x, half_y = room.half()
    add_counter(
        worldbody,
        center_xy=(ox, cy),
        length=COUNTER_LENGTH,
        depth=COUNTER_DEPTH,
        top_z=COUNTER_TOP_Z,
    )
    add_backsplash_and_window(
        worldbody, counter_y=cy, half_x=half_x, wall_y=room.origin_xy[1] + half_y
    )
    add_sink_station(worldbody, cx=ox + SINK_X_LOCAL, cy=cy, top_z=COUNTER_TOP_Z)
    add_stove_station(worldbody, cx=ox + STOVE_X_LOCAL, cy=cy, top_z=COUNTER_TOP_Z)

    cab_x, cab_y = cabinet_origin(room)
    add_cabinet_stub(
        worldbody,
        origin_xy=(cab_x, cab_y),
        depth=CABINET_DEPTH,
        width=CABINET_WIDTH,
        height=CABINET_HEIGHT,
    )
    add_ceiling_lamp_and_switch(
        worldbody,
        switch_pos=(cab_x, cab_y - CABINET_WIDTH / 2.0 - 0.01, 1.25),
    )

    if layout.include_extra_objects and layout.room_id == "kitchen":
        items = list(default_kitchen_small_objects(cy, COUNTER_TOP_Z))
        if layout.cups:
            items = [("cup", spec) for spec in layout.cups] + [
                it for it in items if it[0] != "cup"
            ]
        place_small_objects(worldbody, items, counter_y=cy, top_z=COUNTER_TOP_Z)
    elif layout.cups and layout.room_id == "kitchen":
        place_small_objects(
            worldbody,
            [("cup", spec) for spec in layout.cups],
            counter_y=cy,
            top_z=COUNTER_TOP_Z,
        )


def add_living_interior(worldbody: ET.Element, room: RoomSpec, layout: SceneLayout) -> None:
    ox, oy = room.origin_xy
    add_coffee_table(worldbody, center_xy=(ox + 0.15, oy))
    add_sofa_stub(worldbody, center_xy=(ox + 1.15, oy), yaw=0.0)
    add_floor_lamp(worldbody, pos_xy=(ox + 1.35, oy - 1.35))
    if layout.include_extra_objects and layout.room_id == "living":
        place_small_objects(
            worldbody,
            [
                (
                    "bowl",
                    {
                        "name": "living_bowl",
                        "position": (ox + 0.15, oy + 0.08, 0.42 + 0.035),
                    },
                )
            ],
            counter_y=oy,
            top_z=0.42,
        )


def add_home_world(root: ET.Element, home: HomeSpec, layout: SceneLayout) -> None:
    worldbody = ensure_child(root, "worldbody")
    _add_env_lights(worldbody)
    _add_floor(worldbody, home)
    for room in home.rooms:
        _add_room_shell(worldbody, home, room)
        if room.kind == "kitchen":
            add_kitchen_interior(worldbody, room, layout)
        elif room.kind == "living":
            add_living_interior(worldbody, room, layout)


def apartment_extent(home: HomeSpec) -> tuple[tuple[float, float, float], float]:
    """viewer statistic：(center, extent)。"""
    xs: list[float] = []
    ys: list[float] = []
    for room in home.rooms:
        hx, hy = room.half()
        xs.extend((room.origin_xy[0] - hx, room.origin_xy[0] + hx))
        ys.extend((room.origin_xy[1] - hy, room.origin_xy[1] + hy))
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys)) * 0.25
    span = max(max(xs) - min(xs), max(ys) - min(ys), WALL_HEIGHT)
    return (cx, cy, 1.0), span * 0.85
