"""资产登记表：category / id / 颜色变体 / 文件路径。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from cup_object import CUP_COLORS
from paths import FIXTURES_DIR, OBJECTS_DIR, ROOMS_DIR


@dataclass(frozen=True)
class AssetSpec:
    """单个可拼装资产。"""

    asset_id: str
    category: str
    path: Path | None = None
    colors: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_stub(self) -> bool:
        if "procedural" in self.tags:
            return False
        return self.path is None or not self.path.exists()


@dataclass
class AssetRegistry:
    assets: dict[str, AssetSpec] = field(default_factory=dict)

    def register(self, spec: AssetSpec) -> None:
        if spec.asset_id in self.assets:
            raise ValueError(f"资产 id 已存在: {spec.asset_id}")
        self.assets[spec.asset_id] = spec

    def get(self, asset_id: str) -> AssetSpec:
        try:
            return self.assets[asset_id]
        except KeyError as exc:
            raise KeyError(f"未知资产: {asset_id}") from exc

    def by_category(self, category: str) -> list[AssetSpec]:
        return [a for a in self.assets.values() if a.category == category]

    def list_ids(self, category: str | None = None) -> list[str]:
        specs: Iterable[AssetSpec] = (
            self.by_category(category) if category else self.assets.values()
        )
        return sorted(a.asset_id for a in specs)


def default_registry(*, scan_robocasa: bool = True) -> AssetRegistry:
    """内置 stub + 程序化杯子；可选扫描 assets/robocasa。"""
    reg = AssetRegistry()
    builtins = [
        AssetSpec(
            asset_id="home.apartment_stub",
            category="home",
            path=ROOMS_DIR / "apartment_stub.xml",
            tags=("home", "stub", "procedural"),
            note="厨房+客厅程序化公寓，由 layout.HomeSpec 描述",
        ),
        AssetSpec(
            asset_id="room.kitchen_stub",
            category="room",
            path=ROOMS_DIR / "kitchen_stub.xml",
            tags=("kitchen", "stub", "procedural"),
            note="公寓内厨房；程序化拼装",
        ),
        AssetSpec(
            asset_id="room.living_stub",
            category="room",
            path=ROOMS_DIR / "living_stub.xml",
            tags=("living", "stub", "procedural"),
            note="公寓内客厅最小集：茶几/沙发/落地灯",
        ),
        AssetSpec(
            asset_id="fixture.counter_stub",
            category="fixture",
            path=FIXTURES_DIR / "counter_stub.xml",
            tags=("counter", "stub", "procedural"),
            note="由 fixtures.add_counter 生成",
        ),
        AssetSpec(
            asset_id="fixture.cabinet_stub",
            category="fixture",
            path=FIXTURES_DIR / "cabinet_stub.xml",
            tags=("cabinet", "door", "stub", "procedural"),
            note="程序化铰链柜；RoboCasa 柜门下载后可同 id 替换",
        ),
        AssetSpec(
            asset_id="fixture.coffee_table_stub",
            category="fixture",
            path=FIXTURES_DIR / "coffee_table_stub.xml",
            tags=("table", "living", "stub", "procedural"),
            note="客厅茶几 stub",
        ),
        AssetSpec(
            asset_id="object.cup",
            category="object",
            path=OBJECTS_DIR / "cup_procedural.py",
            colors=tuple(sorted(CUP_COLORS)),
            tags=("graspable", "procedural"),
            note="lib/cup_object.py 程序化彩色杯",
        ),
        AssetSpec(
            asset_id="object.bag",
            category="object",
            path=OBJECTS_DIR / "bag.xml",
            tags=("lift", "stub"),
            note="占位：拎包几何后续自制",
        ),
    ]
    for spec in builtins:
        reg.register(spec)

    if scan_robocasa:
        try:
            from robocasa_bridge import scan_and_register

            scan_and_register(reg)
        except Exception:
            pass
    return reg
