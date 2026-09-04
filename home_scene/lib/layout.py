"""家庭场景规格：Home → Room → Workstation → Episode Layout。

房间相对稳定；Layout 只描述一次 episode（机器人站哪个工位、柜门开多大）。
移动方式目前是工位瞬移：pelvis 仍固定，换工位即换挂载位姿。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from cup_object import CupSpec
from g1_mount import DEFAULT_ROBOT_YAW

# ---------------------------------------------------------------------------
# 尺度（米）：以 G1 ~1.32m、pelvis z≈0.793 为参照
# ---------------------------------------------------------------------------
PELVIS_Z = 0.793
WALL_HEIGHT = 2.60
WALL_THICKNESS = 0.08
DOOR_WIDTH = 1.00
DOOR_HEIGHT = 2.10

COUNTER_TOP_Z = 0.90
COUNTER_DEPTH = 0.60
COUNTER_LENGTH = 4.20
SINK_X_LOCAL = -1.25
STOVE_X_LOCAL = 1.25

CABINET_WIDTH = 0.70
CABINET_DEPTH = 0.36
CABINET_HEIGHT = 1.80

ROBOT_BODY_FORWARD = 0.15
WORKSTATION_STANDOFF = 0.55

KITCHEN_SIZE = (6.0, 5.0)
LIVING_SIZE = (4.0, 5.0)
# 厨房原点 (0,0)；客厅贴东墙，共享 x=3 处门洞
KITCHEN_ORIGIN = (0.0, 0.0)
LIVING_ORIGIN = (5.0, 0.0)

# 兼容旧脚本
ROOM_SIZE_X, ROOM_SIZE_Y = KITCHEN_SIZE


@dataclass(frozen=True)
class Workstation:
    """操作工位：固定 pelvis 的世界位姿（瞬移，不做导航）。"""

    name: str
    position: tuple[float, float, float]
    yaw: float
    note: str = ""


@dataclass(frozen=True)
class Portal:
    """房间之间的门洞。wall 是 from_room 上被切开的墙。"""

    name: str
    from_room: str
    to_room: str
    wall: str  # north / south / east / west
    width: float = DOOR_WIDTH
    height: float = DOOR_HEIGHT
    along: float = 0.0  # 沿墙相对房间中心的偏移


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    kind: str  # kitchen | living | ...
    size_xy: tuple[float, float]
    origin_xy: tuple[float, float]
    wall_height: float = WALL_HEIGHT
    workstations: tuple[Workstation, ...] = ()

    def half(self) -> tuple[float, float]:
        return self.size_xy[0] / 2.0, self.size_xy[1] / 2.0

    def workstation(self, name: str) -> Workstation:
        for ws in self.workstations:
            if ws.name == name:
                return ws
        known = [w.name for w in self.workstations]
        raise KeyError(f"房间 {self.room_id} 无工位 {name}；已有 {known}")


@dataclass(frozen=True)
class HomeSpec:
    home_id: str
    rooms: tuple[RoomSpec, ...]
    portals: tuple[Portal, ...] = ()

    def room(self, room_id: str) -> RoomSpec:
        for room in self.rooms:
            if room.room_id == room_id:
                return room
        known = [r.room_id for r in self.rooms]
        raise KeyError(f"未知房间 {room_id}；已有 {known}")

    def iter_rooms(self) -> Iterable[RoomSpec]:
        return self.rooms


def counter_center_y(room: RoomSpec) -> float:
    _, half_y = room.half()
    return room.origin_xy[1] + half_y - WALL_THICKNESS / 2.0 - COUNTER_DEPTH / 2.0


def counter_front_y(room: RoomSpec) -> float:
    return counter_center_y(room) - COUNTER_DEPTH / 2.0


def cabinet_origin(room: RoomSpec) -> tuple[float, float]:
    half_x, _ = room.half()
    cab_x = room.origin_xy[0] - half_x + WALL_THICKNESS / 2.0 + CABINET_DEPTH / 2.0
    cab_y = room.origin_xy[1] + 0.35
    return cab_x, cab_y


def _stand_off(front: float, *, toward_negative: bool) -> float:
    delta = WORKSTATION_STANDOFF + ROBOT_BODY_FORWARD
    return front - delta if toward_negative else front + delta


def _kitchen_workstations(room_xy: tuple[float, float], size_xy: tuple[float, float]) -> tuple[Workstation, ...]:
    room = RoomSpec(
        room_id="_tmp",
        kind="kitchen",
        size_xy=size_xy,
        origin_xy=room_xy,
    )
    ox, _ = room.origin_xy
    front_y = counter_front_y(room)
    robot_y = _stand_off(front_y, toward_negative=True)
    cab_x, cab_y = cabinet_origin(room)
    door_face_x = cab_x + CABINET_DEPTH / 2.0
    robot_x = _stand_off(door_face_x, toward_negative=False)
    return (
        Workstation(
            "counter_front",
            (ox, robot_y, PELVIS_Z),
            DEFAULT_ROBOT_YAW,
            "面朝北侧台面",
        ),
        Workstation(
            "sink_front",
            (ox + SINK_X_LOCAL, robot_y, PELVIS_Z),
            DEFAULT_ROBOT_YAW,
            "面朝水槽",
        ),
        Workstation(
            "stove_front",
            (ox + STOVE_X_LOCAL, robot_y, PELVIS_Z),
            DEFAULT_ROBOT_YAW,
            "面朝灶台",
        ),
        Workstation(
            "cabinet_west",
            (robot_x, cab_y, PELVIS_Z),
            math.pi,
            "面朝西侧柜门",
        ),
    )


def _living_workstations(origin_xy: tuple[float, float]) -> tuple[Workstation, ...]:
    ox, oy = origin_xy
    table_cx = ox + 0.15
    table_west = table_cx - 0.55  # 与 coffee_table 默认半长一致
    robot_x = _stand_off(table_west, toward_negative=True)
    return (
        Workstation(
            "table_front",
            (robot_x, oy, PELVIS_Z),
            0.0,
            "面朝东侧茶几",
        ),
    )


def make_apartment_stub() -> HomeSpec:
    kitchen = RoomSpec(
        room_id="kitchen",
        kind="kitchen",
        size_xy=KITCHEN_SIZE,
        origin_xy=KITCHEN_ORIGIN,
        workstations=_kitchen_workstations(KITCHEN_ORIGIN, KITCHEN_SIZE),
    )
    living = RoomSpec(
        room_id="living",
        kind="living",
        size_xy=LIVING_SIZE,
        origin_xy=LIVING_ORIGIN,
        workstations=_living_workstations(LIVING_ORIGIN),
    )
    return HomeSpec(
        home_id="home.apartment_stub",
        rooms=(kitchen, living),
        portals=(
            Portal(
                name="kitchen_living_door",
                from_room="kitchen",
                to_room="living",
                wall="east",
                along=0.0,
            ),
        ),
    )


APARTMENT = make_apartment_stub()
HOMES: dict[str, HomeSpec] = {APARTMENT.home_id: APARTMENT}


def get_home(home_id: str) -> HomeSpec:
    try:
        return HOMES[home_id]
    except KeyError as exc:
        raise KeyError(f"未知 Home: {home_id}；已有 {sorted(HOMES)}") from exc


@dataclass(frozen=True)
class SceneLayout:
    """一次 episode：在哪套房子、站哪个工位、夹具/物体初态。"""

    home_id: str = "home.apartment_stub"
    room_id: str = "kitchen"
    workstation: str = "counter_front"
    include_extra_objects: bool = True
    cabinet_open_angle: float = 0.0
    cups: tuple[CupSpec, ...] = ()
    extra_tags: tuple[str, ...] = field(default_factory=tuple)

    def resolve(self) -> tuple[HomeSpec, RoomSpec, Workstation]:
        home = get_home(self.home_id)
        room = home.room(self.room_id)
        return home, room, room.workstation(self.workstation)

    @property
    def robot_position(self) -> tuple[float, float, float]:
        return self.resolve()[2].position

    @property
    def robot_yaw(self) -> float:
        return self.resolve()[2].yaw

    @classmethod
    def kitchen_stub(cls) -> SceneLayout:
        return cls(
            room_id="kitchen",
            workstation="counter_front",
            include_extra_objects=True,
        )

    @classmethod
    def close_cabinet(cls) -> SceneLayout:
        return cls(
            room_id="kitchen",
            workstation="cabinet_west",
            include_extra_objects=False,
            cabinet_open_angle=0.9,
            extra_tags=("task:home_close_cabinet",),
        )

    @classmethod
    def living_stub(cls) -> SceneLayout:
        return cls(
            room_id="living",
            workstation="table_front",
            include_extra_objects=True,
            extra_tags=("room:living",),
        )


LAYOUT_FACTORIES = {
    "kitchen": SceneLayout.kitchen_stub,
    "kitchen_stub": SceneLayout.kitchen_stub,
    "default": SceneLayout.kitchen_stub,
    "close_cabinet": SceneLayout.close_cabinet,
    "cabinet": SceneLayout.close_cabinet,
    "living": SceneLayout.living_stub,
    "living_stub": SceneLayout.living_stub,
}

LAYOUT_CHOICES = ("kitchen", "close_cabinet", "living")


def layout_from_name(name: str) -> SceneLayout:
    factory = LAYOUT_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"未知 layout: {name}；可选 {sorted(set(LAYOUT_CHOICES))}")
    return factory()


# 旧名：默认厨房工位
def default_robot_pose() -> tuple[tuple[float, float, float], float]:
    ws = APARTMENT.room("kitchen").workstation("counter_front")
    return ws.position, ws.yaw


def close_cabinet_robot_pose() -> tuple[tuple[float, float, float], float]:
    ws = APARTMENT.room("kitchen").workstation("cabinet_west")
    return ws.position, ws.yaw
