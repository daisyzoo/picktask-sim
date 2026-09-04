#!/usr/bin/env python3
"""打开 home_scene viewer（默认同开头显预览）。

macOS:
  mjpython scripts/view_scene.py
  mjpython scripts/view_scene.py --layout close_cabinet
  mjpython scripts/view_scene.py --layout living
  mjpython scripts/view_scene.py --no-preview
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
from g1_mount import ACTUATED_JOINTS, sync_g1_ctrl_to_qpos  # noqa: E402
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


def _require_mjpython_on_macos() -> None:
    if platform.system() != "Darwin":
        return
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return
    print(
        "错误: macOS 上 launch_passive 必须用 mjpython 启动。\n"
        "  mjpython scripts/view_scene.py [--layout kitchen|close_cabinet|living]",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        default="kitchen",
        choices=LAYOUT_CHOICES,
        help="工位布局（同一套公寓，只换 pelvis 挂载点）",
    )
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    _require_mjpython_on_macos()
    assert_decoupled()
    layout = layout_from_name(args.layout)
    model, data = make_model_data(layout)
    renderer = make_head_renderer(model)
    if not args.no_preview:
        init_head_camera_preview(enable_by_default=True)

    print(
        f"home_scene viewer · layout={args.layout} "
        f"nq={model.nq} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}"
    )
    print(
        f"G1 全自由度: {len(ACTUATED_JOINTS)} 关节（pelvis 固定 · "
        f"{layout.room_id}/{layout.workstation}）。"
    )
    print("仅观测；要操作请用: mjpython scripts/teleop.py")
    print("若看不到窗口，Cmd+Tab 切到 mjpython。")

    sync_g1_ctrl_to_qpos(model, data)
    try:
        with mujoco.viewer.launch_passive(model, data, show_right_ui=False) as viewer:
            while viewer.is_running():
                step_start = time.time()
                rgb = None
                with viewer.lock():
                    sync_g1_ctrl_to_qpos(model, data)
                    mujoco.mj_step(model, data)
                    sync_appliances(model, data)
                    rgb = render_head_camera(renderer, data)
                    configure_viewer(viewer)
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
