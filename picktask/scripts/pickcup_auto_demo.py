#!/usr/bin/env python3
"""自动抓杯演示：启动 MuJoCo 仿真，机器人 scripted 抓取杯子并录制 LeRobot dataset。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from _bootstrap import bootstrap

bootstrap()

import pickcup as pc
from lerobot_recorder import CAMERA_HEIGHT, CAMERA_WIDTH, DEFAULT_TASK, PickcupLeRobotRecorder, RECORD_FPS
from pickcup_sim_core import AutoPickController, SimContext, check_episode_success, run_scripted_episode


def print_banner(session_root: Path) -> None:
    print("=" * 60)
    print("MuJoCo G1 自动抓杯演示（LeRobot 录制）")
    print("=" * 60)
    print(f"  - 任务: {DEFAULT_TASK}")
    print(f"  - 录制: head_camera {RECORD_FPS}fps -> {session_root}")
    print("  - 流程: 就位 -> 接近 -> 闭合 -> 抬起 -> 保持 -> 保存退出")
    print("  - ESC: 提前关闭 viewer 并保存")
    print("=" * 60)


def run_headless(args: argparse.Namespace) -> int:
    scene = pc.SceneLayout.default(cup_variant=args.cup)
    ctx = SimContext.from_scene(scene)
    renderer = mujoco.Renderer(ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
    recorder = PickcupLeRobotRecorder(
        pc.SESSIONS_DATA_DIR,
        pc.PROPRIO_STATE_NAMES,
        pc.ACTION_NAMES,
        task=DEFAULT_TASK,
    )
    controller = AutoPickController()
    session_root = recorder.start_session()
    print_banner(session_root)
    print(f"  - 杯子变体: {scene.cup_variant}")

    try:
        success = run_scripted_episode(ctx, controller, renderer, recorder, record=True)
        if success:
            recorder.save_episode()
        else:
            recorder.discard_episode()
    finally:
        recorder.close_session()

    print(f"\n演示结束 success={success}, dataset: {session_root}")
    return 0 if success else 1


def run_with_viewer(args: argparse.Namespace) -> int:
    scene = pc.SceneLayout.default(cup_variant=args.cup)
    ctx = SimContext.from_scene(scene)
    renderer = mujoco.Renderer(ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
    recorder = PickcupLeRobotRecorder(
        pc.SESSIONS_DATA_DIR,
        pc.PROPRIO_STATE_NAMES,
        pc.ACTION_NAMES,
        task=DEFAULT_TASK,
    )
    controller = AutoPickController()
    should_exit = False
    next_record_time = 0.0

    import glfw

    def key_callback(key: int | str) -> None:
        nonlocal should_exit
        key = pc.resolve_key(key)
        if key == glfw.KEY_ESCAPE:
            should_exit = True
            print(">>> 正在退出 viewer")

    controller.reset(
        {joint: ctx.robot_ctrl[joint] for joint in pc.KEY_CONTROL_JOINTS},
        scene,
    )
    controller.bind_initial_cup_z(ctx.initial_cup_z())
    session_root = recorder.start_session()
    preview_enabled = pc.init_head_camera_preview() if not args.no_preview else False
    print_banner(session_root)
    print(f"  - 杯子变体: {scene.cup_variant}")

    exit_code = 1
    try:
        with mujoco.viewer.launch_passive(
            ctx.model,
            ctx.data,
            key_callback=key_callback,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running() and not should_exit and not controller.finished:
                loop_start = time.time()
                preview_rgb: np.ndarray | None = None

                with viewer.lock():
                    controller.advance(
                        ctx.model.opt.timestep,
                        ctx.model,
                        ctx.data,
                        ctx.robot_ctrl,
                        ctx.grasp,
                        ctx.teleop_geoms,
                        ctx.obstacle_geoms,
                        ctx.gripper_geoms,
                        ctx.cup_geoms,
                    )
                    ctx.step()
                    next_record_time, preview_rgb = pc.maybe_record_frame(
                        ctx.model,
                        ctx.data,
                        ctx.robot_ctrl,
                        renderer,
                        recorder,
                        next_record_time,
                    )
                    pc._configure_viewer(viewer)

                viewer.sync()

                if preview_enabled and preview_rgb is not None:
                    pc.show_head_camera_preview(preview_rgb)

                sleep_time = ctx.model.opt.timestep - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            success = check_episode_success(
                controller.metrics,
                ctx.model,
                ctx.data,
                ctx.robot_ctrl,
                ctx.grasp,
                ctx.gripper_geoms,
                ctx.cup_geoms,
            )
            exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        print("\n\n正在保存 LeRobot dataset...")
        if recorder.is_active:
            if exit_code == 0 and recorder.dataset and recorder.dataset.has_pending_frames():
                recorder.save_episode()
            elif recorder.dataset and recorder.dataset.has_pending_frames():
                recorder.discard_episode()
            recorder.close_session()
        if preview_enabled:
            pc.close_head_camera_preview()

    print(f"\n演示结束 success={exit_code == 0}, dataset: {session_root}")
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 自动抓杯演示 + LeRobot 录制")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    pc.add_cup_cli_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        raise SystemExit(run_headless(args))
    raise SystemExit(run_with_viewer(args))


if __name__ == "__main__":
    main()
