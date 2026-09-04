"""程序生成光滑截锥杯 mesh（视觉用），碰撞仍由圆柱分段承担。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paths import CACHE_DIR

CUP_MESH_CACHE_DIR = CACHE_DIR / "cup_meshes"


@dataclass(frozen=True)
class CupVisualProfile:
    """杯体视觉轮廓（单位：米，相对 cup body 原点，底面约在 z=0）。"""

    bottom_radius: float
    top_radius: float
    height: float
    wall_thickness: float = 0.0015
    n_theta: int = 64
    n_height: int = 32


def profile_from_segments(segments: tuple) -> CupVisualProfile:
    """从碰撞圆柱分段推断视觉截锥轮廓。"""
    bottom_r = float(segments[0][1])
    top_r = float(segments[-1][1])
    # 分段中心 z ± half_height → 杯底约 0，杯口约最高点。
    z0 = float(segments[0][5]) - float(segments[0][2])
    z1 = float(segments[-1][5]) + float(segments[-1][2])
    height = max(z1 - z0, 1e-4)
    # 视觉 mesh 从 body 局部 z=0 起算；碰撞分段已按桌面放置。
    return CupVisualProfile(
        bottom_radius=bottom_r,
        top_radius=top_r,
        height=height,
        wall_thickness=0.0015 if top_r > 0.038 else 0.0012,
    )


def _ring_points(radius: float, z: float, n_theta: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    return np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(n_theta, z, dtype=np.float64),
        )
    )


def _append_side(
    vertices: list[np.ndarray],
    faces: list[tuple[int, int, int]],
    *,
    bottom_r: float,
    top_r: float,
    z0: float,
    z1: float,
    n_theta: int,
    n_height: int,
    outward: bool,
) -> None:
    """追加截锥侧壁（三角形扇）。outward=True 外法向朝外。"""
    base = len(vertices)
    for i in range(n_height + 1):
        t = i / n_height
        r = bottom_r + t * (top_r - bottom_r)
        z = z0 + t * (z1 - z0)
        for point in _ring_points(r, z, n_theta):
            vertices.append(point)

    for i in range(n_height):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a = base + i * n_theta + j
            b = base + i * n_theta + j2
            c = base + (i + 1) * n_theta + j2
            d = base + (i + 1) * n_theta + j
            if outward:
                faces.append((a, b, c))
                faces.append((a, c, d))
            else:
                faces.append((a, c, b))
                faces.append((a, d, c))


def _append_annulus(
    vertices: list[np.ndarray],
    faces: list[tuple[int, int, int]],
    *,
    inner_r: float,
    outer_r: float,
    z: float,
    n_theta: int,
    normal_up: bool,
) -> None:
    base = len(vertices)
    for point in _ring_points(inner_r, z, n_theta):
        vertices.append(point)
    for point in _ring_points(outer_r, z, n_theta):
        vertices.append(point)
    for j in range(n_theta):
        j2 = (j + 1) % n_theta
        i0 = base + j
        i1 = base + j2
        o0 = base + n_theta + j
        o1 = base + n_theta + j2
        if normal_up:
            faces.append((i0, o0, o1))
            faces.append((i0, o1, i1))
        else:
            faces.append((i0, o1, o0))
            faces.append((i0, i1, o1))


def _append_disk(
    vertices: list[np.ndarray],
    faces: list[tuple[int, int, int]],
    *,
    radius: float,
    z: float,
    n_theta: int,
    normal_up: bool,
) -> None:
    base = len(vertices)
    vertices.append(np.array([0.0, 0.0, z], dtype=np.float64))
    for point in _ring_points(radius, z, n_theta):
        vertices.append(point)
    center = base
    for j in range(n_theta):
        j2 = (j + 1) % n_theta
        a = base + 1 + j
        b = base + 1 + j2
        if normal_up:
            faces.append((center, a, b))
        else:
            faces.append((center, b, a))


def build_cup_mesh_parts(profile: CupVisualProfile) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """生成外/内壁、杯底、口沿四块三角 mesh。"""
    outer_bottom = profile.bottom_radius
    outer_top = profile.top_radius
    thick = profile.wall_thickness
    inner_bottom = max(outer_bottom - thick, 0.004)
    inner_top = max(outer_top - thick, inner_bottom + 0.002)
    z0 = 0.0
    z1 = profile.height
    # 内底略高于外底，形成杯底厚度。
    bottom_thickness = min(0.003, profile.height * 0.05)
    z_inner_floor = z0 + bottom_thickness

    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 外壁
    v: list[np.ndarray] = []
    f: list[tuple[int, int, int]] = []
    _append_side(
        v,
        f,
        bottom_r=outer_bottom,
        top_r=outer_top,
        z0=z0,
        z1=z1,
        n_theta=profile.n_theta,
        n_height=profile.n_height,
        outward=True,
    )
    parts["outer"] = (np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32))

    # 内壁（从上到下同向参数，法向朝内）
    v, f = [], []
    _append_side(
        v,
        f,
        bottom_r=inner_bottom,
        top_r=inner_top,
        z0=z_inner_floor,
        z1=z1,
        n_theta=profile.n_theta,
        n_height=profile.n_height,
        outward=False,
    )
    parts["inner"] = (np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32))

    # 杯底：外底面朝下 + 内底面朝上 + 底侧壁环
    v, f = [], []
    _append_disk(v, f, radius=outer_bottom, z=z0, n_theta=profile.n_theta, normal_up=False)
    _append_disk(v, f, radius=inner_bottom, z=z_inner_floor, n_theta=profile.n_theta, normal_up=True)
    _append_side(
        v,
        f,
        bottom_r=outer_bottom,
        top_r=inner_bottom,
        z0=z0,
        z1=z_inner_floor,
        n_theta=profile.n_theta,
        n_height=2,
        outward=True,
    )
    parts["bottom"] = (np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32))

    # 口沿环面
    v, f = [], []
    _append_annulus(
        v,
        f,
        inner_r=inner_top,
        outer_r=outer_top,
        z=z1,
        n_theta=profile.n_theta,
        normal_up=True,
    )
    parts["rim"] = (np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32))

    return parts


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# generated by picktask.cup_geometry", f"# vertices {len(vertices)} faces {len(faces)}"]
    for x, y, z in vertices:
        lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
    for a, b, c in faces:
        # OBJ 为 1-based
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_cup_visual_meshes(
    variant: str,
    profile: CupVisualProfile,
) -> dict[str, Path]:
    """确保缓存目录中有该变体的视觉 OBJ，返回 part->path。"""
    out_dir = CUP_MESH_CACHE_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = build_cup_mesh_parts(profile)
    paths: dict[str, Path] = {}
    for name, (vertices, faces) in parts.items():
        path = out_dir / f"cup_{name}.obj"
        # 轮廓变化时重写；用简单 marker 文件记录参数。
        marker = out_dir / f"cup_{name}.meta"
        meta = (
            f"{profile.bottom_radius:.6f},{profile.top_radius:.6f},"
            f"{profile.height:.6f},{profile.wall_thickness:.6f},"
            f"{profile.n_theta},{profile.n_height}"
        )
        if not path.exists() or not marker.exists() or marker.read_text() != meta:
            write_obj(path, vertices, faces)
            marker.write_text(meta, encoding="utf-8")
        paths[name] = path.resolve()
    return paths
