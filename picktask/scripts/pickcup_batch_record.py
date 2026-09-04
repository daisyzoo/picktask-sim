#!/usr/bin/env python3
"""批量 scripted 抓杯并写入单个 LeRobot dataset（随机机器人/杯子位置）。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import bootstrap

bootstrap()

import pickcup as pc
from lerobot_recorder import CAMERA_HEIGHT, CAMERA_WIDTH, DEFAULT_TASK, PickcupLeRobotRecorder
from pickcup_sim_core import (
    AutoPickController,
    SUCCESS_CRITERIA_DESCRIPTION,
    SimContext,
    run_scripted_episode,
)

DEFAULT_OUTPUT = "pickcup_train"
NUM_APPROACH_VARIANTS = 4
PREVIEW_CAMERA = 96


def find_working_variant(scene: pc.SceneLayout) -> int | None:
    """无录制快速试跑 4 种 approach variant，返回首个成功 variant。"""
    for variant in range(NUM_APPROACH_VARIANTS):
        ctx = SimContext.from_scene(scene)
        renderer = mujoco.Renderer(ctx.model, PREVIEW_CAMERA, PREVIEW_CAMERA)
        controller = AutoPickController()
        controller.variant_index = variant
        if run_scripted_episode(
            ctx,
            controller,
            renderer,
            recorder=None,
            record=False,
            rng=None,
        ):
            return variant
    return None


def try_record_one_episode(
    scene: pc.SceneLayout,
    recorder: PickcupLeRobotRecorder,
    rng: np.random.Generator,
) -> tuple[bool, float, int | None]:
    t0 = time.time()
    variant = find_working_variant(scene)
    if variant is None:
        return False, time.time() - t0, None

    ctx = SimContext.from_scene(scene)
    renderer = mujoco.Renderer(ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
    controller = AutoPickController()
    controller.variant_index = variant
    success = run_scripted_episode(
        ctx,
        controller,
        renderer,
        recorder,
        record=True,
        rng=None,
    )
    elapsed = time.time() - t0
    if success:
        recorder.save_episode()
    else:
        recorder.discard_episode()
    return success, elapsed, variant


def batch_record(
    *,
    target_episodes: int,
    max_attempts: int,
    output_name: str,
    seed: int,
    cup_variant: str | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    variant = pc.resolve_cup_variant(cup_variant)
    recorder = PickcupLeRobotRecorder(
        pc.TRAIN_DATA_DIR,
        pc.PROPRIO_STATE_NAMES,
        pc.ACTION_NAMES,
        task=DEFAULT_TASK,
        session_name=output_name,
    )
    session_root = recorder.start_session()
    print("=" * 60)
    print("批量 scripted 抓杯录制")
    print("=" * 60)
    print(f"  目标成功 episode: {target_episodes}")
    print(f"  最大尝试次数: {max_attempts}")
    print(f"  杯子变体: {variant}")
    print(f"  输出: {session_root}")
    print("=" * 60)
    print(f"  成功标准: {SUCCESS_CRITERIA_DESCRIPTION}")
    saved = 0
    attempts = 0
    while saved < target_episodes and attempts < max_attempts:
        attempts += 1
        scene = pc.sample_random_scene(rng, cup_variant=variant)
        success, elapsed, variant_idx = try_record_one_episode(scene, recorder, rng)
        if success:
            saved += 1
        status = "OK" if success else "FAIL"
        variant_text = f"var={variant_idx}" if variant_idx is not None else "var=-"
        print(
            f"[{attempts:4d}] {status} saved={saved}/{target_episodes} "
            f"{variant_text} t={elapsed:.1f}s "
            f"robot=({scene.robot_position[0]:.2f},{scene.robot_position[1]:.2f}) "
            f"cup=({scene.cup_position[0]:.2f},{scene.cup_position[1]:.2f},yaw={scene.cup_yaw:.2f})"
        )

    recorder.close_session()
    print(f"\n完成: {saved}/{target_episodes} episodes -> {session_root}")
    if saved < target_episodes:
        print(f"警告: 未达目标，仅保存 {saved} 条（尝试 {attempts} 次）")
    return session_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量 scripted 抓杯 LeRobot 录制")
    parser.add_argument("--episodes", type=int, default=20, help="目标成功 episode 数")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="最大尝试次数（0 表示 episodes 的 15 倍）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"dataset 目录名（位于 pickcupdata/train/，默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument("--seed", type=int, default=42)
    pc.add_cup_cli_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_attempts = args.max_attempts or max(args.episodes * 15, args.episodes + 20)
    batch_record(
        target_episodes=args.episodes,
        max_attempts=max_attempts,
        output_name=args.output,
        seed=args.seed,
        cup_variant=args.cup,
    )


if __name__ == "__main__":
    main()
