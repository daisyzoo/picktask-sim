#!/usr/bin/env python3
"""home_scene 键盘遥操作（可操作机器人 + 头显预览 + 家电）。

macOS:
  mjpython scripts/teleop.py
  mjpython scripts/teleop.py --layout close_cabinet
  mjpython scripts/teleop.py --layout living
  HOME_SCENE_HEAD_CAMERA_PREVIEW=0 mjpython scripts/teleop.py   # 关预览
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from appliances import sync_appliances  # noqa: E402
from head_camera import (  # noqa: E402
    close_head_camera_preview,
    init_head_camera_preview,
    make_head_renderer,
    render_head_camera,
    show_head_camera_preview,
)
from paths import assert_decoupled  # noqa: E402
from layout import LAYOUT_CHOICES, layout_from_name  # noqa: E402
from scene_composer import configure_viewer, make_model_data  # noqa: E402
from teleop import HELP_TEXT, TeleopState, make_key_callback  # noqa: E402


def _require_mjpython_on_macos() -> None:
    if platform.system() != "Darwin":
        return
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return
    print(
        "错误: macOS 上必须用 mjpython 启动。\n"
        "  mjpython scripts/teleop.py [--layout kitchen|close_cabinet|living]",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        default="kitchen",
        choices=LAYOUT_CHOICES,
        help="工位布局（同一套公寓，只换 pelvis 挂载点）",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="关闭头显预览窗口",
    )
    args = parser.parse_args()

    _require_mjpython_on_macos()
    assert_decoupled()
    layout = layout_from_name(args.layout)
    model, data = make_model_data(layout)
    state = TeleopState(model, data)
    renderer = make_head_renderer(model)
    if not args.no_preview:
        init_head_camera_preview(enable_by_default=True)

    print(
        f"home_scene teleop · layout={args.layout} "
        f"{layout.room_id}/{layout.workstation} "
        f"nq={model.nq} nu={model.nu}  G1_actuators=23"
    )
    print(HELP_TEXT)
    print("若看不到窗口，Cmd+Tab 切到 mjpython。点击 3D 窗口后再按键。")

    key_cb = make_key_callback(state)
    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_cb, show_right_ui=False
        ) as viewer:
            while viewer.is_running() and not state.quit:
                step_start = time.time()
                rgb = None
                with viewer.lock():
                    state.apply()
                    mujoco.mj_step(model, data)
                    sync_appliances(model, data)
                    rgb = render_head_camera(renderer, data)
                    configure_viewer(viewer)
                state.maybe_print()
                if rgb is not None:
                    show_head_camera_preview(rgb)
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    finally:
        close_head_camera_preview()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
