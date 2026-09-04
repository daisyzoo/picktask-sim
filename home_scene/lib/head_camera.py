"""home_scene 头相机渲染与预览（子进程 OpenCV，避免 mjpython 冲突）。"""

from __future__ import annotations

import os
from typing import Sequence

import mujoco
import numpy as np

from g1_mount import (
    D435_COLOR_CX,
    D435_COLOR_CY,
    D435_COLOR_FX,
    D435_COLOR_FY,
    D435_COLOR_RESOLUTION,
    HEAD_CAMERA_NAME,
)
from head_camera_preview import HeadCameraPreviewProcess

import math

_HFOV = math.degrees(
    2.0 * math.atan(D435_COLOR_RESOLUTION[0] / (2.0 * D435_COLOR_FX))
)
_VFOV = math.degrees(
    2.0 * math.atan(D435_COLOR_RESOLUTION[1] / (2.0 * D435_COLOR_FY))
)

_preview: HeadCameraPreviewProcess | None = None
_preview_enabled = False


def overlay_lines() -> tuple[str, ...]:
    return (
        "home_scene · D435i head_camera",
        f"{D435_COLOR_RESOLUTION[0]}x{D435_COLOR_RESOLUTION[1]}"
        f"  FOV {_HFOV:.1f} x {_VFOV:.1f} deg",
        f"fx={D435_COLOR_FX:.1f} fy={D435_COLOR_FY:.1f}"
        f"  cx={D435_COLOR_CX:.1f} cy={D435_COLOR_CY:.1f}",
    )


def render_head_camera(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    renderer.update_scene(data, camera=HEAD_CAMERA_NAME)
    return renderer.render()


def make_head_renderer(model: mujoco.MjModel) -> mujoco.Renderer:
    w, h = D435_COLOR_RESOLUTION
    return mujoco.Renderer(model, height=h, width=w)


def init_head_camera_preview(*, enable_by_default: bool = True) -> bool:
    """环境变量 HOME_SCENE_HEAD_CAMERA_PREVIEW=0 可关闭。"""
    global _preview, _preview_enabled
    default = "1" if enable_by_default else "0"
    flag = os.environ.get("HOME_SCENE_HEAD_CAMERA_PREVIEW", default)
    if flag not in {"1", "true", "TRUE", "yes", "YES"}:
        _preview_enabled = False
        print("头显预览已禁用；设置 HOME_SCENE_HEAD_CAMERA_PREVIEW=1 开启")
        return False

    if _preview is not None and _preview.is_alive:
        _preview_enabled = True
        return True

    w, h = D435_COLOR_RESOLUTION
    preview = HeadCameraPreviewProcess(
        width=w,
        height=h,
        window_name="robot_view · D435i",
        overlay_lines=overlay_lines(),
    )
    if not preview.start():
        _preview = None
        _preview_enabled = False
        print("头显预览启动失败（需普通 python + opencv）")
        return False
    _preview = preview
    _preview_enabled = True
    print("头显预览已开启（独立窗口；关窗不影响 3D Viewer）")
    return True


def show_head_camera_preview(rgb: np.ndarray) -> None:
    global _preview, _preview_enabled
    if not _preview_enabled or _preview is None:
        return
    if not _preview.publish(rgb):
        _preview_enabled = False
        _preview.close()
        _preview = None
        print("头显预览已关闭")


def close_head_camera_preview() -> None:
    global _preview, _preview_enabled
    if _preview is not None:
        _preview.close()
    _preview = None
    _preview_enabled = False
