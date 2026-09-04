#!/usr/bin/env python3
"""home_close_cabinet 成功判定冒烟：打开→关闭→保持。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

import mujoco  # noqa: E402

from scene_composer import SceneLayout, make_model_data  # noqa: E402
from tasks.close_cabinet import CloseCabinetTask  # noqa: E402
from g1_mount import sync_g1_ctrl_to_qpos  # noqa: E402


def main() -> int:
    model, data = make_model_data(SceneLayout.close_cabinet())
    task = CloseCabinetTask()
    task.reset(model, data)
    open_ang = task.hinge_angle(data)
    print(f"reset open angle={open_ang:.3f} rad  success_instant={task.success_instant(model, data)}")
    if task.success_instant(model, data):
        print("FAIL: 打开后不应立刻成功")
        return 1

    # 直接合页到闭合
    task.set_hinge_angle(model, data, 0.0)
    if not task.success_instant(model, data):
        print("FAIL: 闭合角应满足 instant 成功")
        return 1

    # 保持计时：用真实 sleep 验证 hold
    task._closed_since = None
    ok = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < task.hold_seconds + 0.3:
        sync_g1_ctrl_to_qpos(model, data)
        mujoco.mj_step(model, data)
        if task.success(model, data):
            ok = True
            break
        time.sleep(0.02)

    print(f"hold success={ok}  angle={task.hinge_angle(data):.4f}")
    if not ok:
        print("FAIL: 保持时间未达成成功")
        return 1

    # 再打开应失败
    task.set_hinge_angle(model, data, 0.5)
    if task.success(model, data):
        print("FAIL: 打开后门应失败")
        return 1

    print("OK: home_close_cabinet success criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
