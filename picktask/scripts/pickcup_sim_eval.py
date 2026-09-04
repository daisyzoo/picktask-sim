#!/usr/bin/env python3
"""在 MuJoCo 中闭环评估 ACT 策略（图像 + proprio -> 关节目标）。"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from _bootstrap import bootstrap

bootstrap(include_lerobot=True)

if TYPE_CHECKING:
    import mujoco

    import pickcup as pc
    from pickcup_sim_core import EpisodeMetrics, SimContext


def pick_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_policy_and_processors(
    policy_path: Path,
    dataset_root: Path,
    device_name: str,
):
    import json

    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    device = pick_device(device_name)

    config_path = policy_path / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict.setdefault("type", "act")
    config_path.write_text(json.dumps(config_dict, indent=4) + "\n")

    # macOS: 必须先 from_pretrained，再 import LeRobotDatasetMetadata，否则可能 segfault。
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    # checkpoint config 里常是 device=cpu；与实际推理设备对齐，避免 batch 在 CPU、权重在 MPS。
    policy.config.device = device.type
    policy.to(device)

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(
        repo_id=f"local/{dataset_root.name}",
        root=dataset_root,
    )
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )
    return policy, metadata, preprocess, postprocess, build_inference_frame, make_robot_action, device


def default_success_log_path(dataset_root: Path) -> Path | None:
    """按批量录制脚本的命名约定，寻找对应的成功场景日志。"""
    if len(dataset_root.parents) < 3:
        return None
    candidate = dataset_root.parents[2] / f"{dataset_root.name}.log"
    return candidate if candidate.exists() else None


def load_success_scenes(
    log_path: Path,
    pc_module,
    *,
    cup_variant: str | None = None,
) -> list[tuple[object, str]]:
    """从 batch record 日志中还原已保存成功 episode 的场景分布。"""
    pattern = re.compile(
        r"\[\s*(?P<attempt>\d+)\]\s+OK .*?"
        r"robot=\((?P<rx>[-0-9.]+),(?P<ry>[-0-9.]+)\) "
        r"cup=\((?P<cx>[-0-9.]+),(?P<cy>[-0-9.]+),yaw=(?P<yaw>[-0-9.]+)\)"
    )
    variant = pc_module.resolve_cup_variant(cup_variant)
    scenes: list[tuple[object, str]] = []
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        scene = pc_module.SceneLayout(
            robot_position=(
                float(match.group("rx")),
                # 旧 success log 可能来自桌边净空不足的场景。保证前脚/小腿
                # 到桌边至少保持当前配置要求的 15 cm。
                min(
                    float(match.group("ry")),
                    pc_module.DEFAULT_ROBOT_POSITION[1],
                ),
                pc_module.DEFAULT_ROBOT_POSITION[2],
            ),
            cup_position=(
                float(match.group("cx")),
                float(match.group("cy")),
                pc_module.DEFAULT_CUP_POSITION[2],
            ),
            cup_yaw=float(match.group("yaw")),
            robot_yaw=pc_module.DEFAULT_ROBOT_YAW,
            cup_variant=variant,
        )
        label = (
            f"log_attempt={int(match.group('attempt'))} "
            f"robot=({scene.robot_position[0]:.2f},{scene.robot_position[1]:.2f}) "
            f"cup=({scene.cup_position[0]:.2f},{scene.cup_position[1]:.2f},yaw={scene.cup_yaw:.2f})"
        )
        scenes.append((scene, label))
    return scenes


def build_eval_scenes(
    *,
    pc_module,
    dataset_root: Path,
    scenes_log: Path | None,
    scene_source: str,
    episodes: int,
    seed: int,
    cup_variant: str | None = None,
) -> tuple[list[tuple[object, str]], str]:
    """构造 eval 场景；默认优先使用训练集中筛选出的成功场景。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    variant = pc_module.resolve_cup_variant(cup_variant)
    if scene_source in {"auto", "success-log"}:
        log_path = scenes_log or default_success_log_path(dataset_root)
        if log_path is not None and log_path.exists():
            scenes = load_success_scenes(log_path, pc_module, cup_variant=variant)
            if scenes:
                order = rng.permutation(len(scenes))
                selected = [scenes[int(order[i % len(order)])] for i in range(episodes)]
                return selected, f"success-log ({log_path}, {len(scenes)} scenes)"
        if scene_source == "success-log":
            raise FileNotFoundError(
                f"未找到可用成功场景日志: {log_path or '<未指定>'}；"
                "请传 --scenes-log 或改用 --scene-source random"
            )

    scenes = []
    for index in range(episodes):
        scene = pc_module.sample_random_scene(rng, cup_variant=variant)
        label = (
            f"random_index={index} "
            f"robot=({scene.robot_position[0]:.2f},{scene.robot_position[1]:.2f}) "
            f"cup=({scene.cup_position[0]:.2f},{scene.cup_position[1]:.2f},yaw={scene.cup_yaw:.2f})"
        )
        scenes.append((scene, label))
    return scenes, "random"


def _show_camera_frame(rgb, *, window_name: str = "pickcup_sim_eval") -> None:
    try:
        import cv2
    except ImportError:
        return
    try:
        cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    except Exception:
        pass


def _close_camera_window() -> None:
    try:
        import cv2

        cv2.destroyAllWindows()
    except Exception:
        pass


def run_policy_episode(
    ctx: SimContext,
    renderer: mujoco.Renderer,
    policy,
    metadata,
    preprocess,
    postprocess,
    build_inference_frame,
    make_robot_action,
    device: object,
    *,
    max_steps: int,
    control_hz: float,
    action_names: tuple[str, ...],
    proprio_state_names: tuple[str, ...],
    default_task: str,
    render: bool = False,
    viewer: object | None = None,
    realtime: bool = False,
) -> tuple[bool, EpisodeMetrics]:
    from pickcup_sim_core import (
        EpisodeMetrics,
        check_episode_success,
        update_episode_grasp_metrics,
    )

    import numpy as np

    import pickcup as pc

    ctx.reset()
    policy.reset()
    metrics = EpisodeMetrics()
    metrics.initial_cup_z = ctx.initial_cup_z()
    control_dt = 1.0 / control_hz
    sim_dt = ctx.model.opt.timestep
    steps_per_control = max(1, int(round(control_dt / sim_dt)))
    latest_rgb = None

    for step in range(max_steps):
        loop_start = time.time()

        def _control_and_step() -> None:
            nonlocal latest_rgb
            if step % steps_per_control == 0:
                latest_rgb = ctx.render_rgb(renderer)
                proprio = ctx.proprio()
                obs_dict = {
                    "head_camera": latest_rgb,
                    **{
                        name: float(proprio[index])
                        for index, name in enumerate(proprio_state_names)
                    },
                }
                obs_frame = build_inference_frame(
                    observation=obs_dict,
                    device=device,
                    ds_features=metadata.features,
                    task=default_task,
                )
                batch = preprocess(obs_frame)
                action_tensor = policy.select_action(batch)
                action_tensor = postprocess(action_tensor)
                action_dict = make_robot_action(action_tensor, metadata.features)
                action_vec = np.array(
                    [action_dict[name] for name in action_names],
                    dtype=np.float32,
                )
                ctx.apply_action_vector(action_vec)

            ctx.step()
            update_episode_grasp_metrics(
                metrics,
                ctx.model,
                ctx.data,
                ctx.robot_ctrl,
                ctx.grasp,
                ctx.gripper_geoms,
                ctx.cup_geoms,
                sim_dt=sim_dt,
                initial_cup_z=metrics.initial_cup_z,
            )

        if viewer is not None:
            with viewer.lock():
                _control_and_step()
                pc._configure_viewer(viewer)
            viewer.sync()
            if not viewer.is_running():
                break
        else:
            _control_and_step()

        if render and latest_rgb is not None and step % steps_per_control == 0:
            _show_camera_frame(latest_rgb)

        if realtime:
            sleep_time = sim_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if check_episode_success(
            metrics,
            ctx.model,
            ctx.data,
            ctx.robot_ctrl,
            ctx.grasp,
            ctx.gripper_geoms,
            ctx.cup_geoms,
        ):
            return True, metrics

    success = check_episode_success(
        metrics,
        ctx.model,
        ctx.data,
        ctx.robot_ctrl,
        ctx.grasp,
        ctx.gripper_geoms,
        ctx.cup_geoms,
    )
    return success, metrics


def evaluate(
    policy_path: Path,
    dataset_root: Path,
    *,
    episodes: int,
    seed: int,
    device_name: str,
    max_steps: int,
    scene_source: str,
    scenes_log: Path | None,
    render: bool = False,
    use_viewer: bool = False,
    realtime: bool | None = None,
    cup_variant: str | None = None,
) -> float:
    (
        policy,
        metadata,
        preprocess,
        postprocess,
        build_inference_frame,
        make_robot_action,
        device,
    ) = load_policy_and_processors(policy_path, dataset_root, device_name)

    # MuJoCo 也必须在 ACT 加载之后再 import。
    import mujoco

    import pickcup as pc
    from lerobot_recorder import CAMERA_HEIGHT, CAMERA_WIDTH, DEFAULT_TASK, RECORD_FPS
    from pickcup_sim_core import SUCCESS_CRITERIA_DESCRIPTION, SimContext

    variant = pc.resolve_cup_variant(cup_variant)
    eval_scenes, scene_source_label = build_eval_scenes(
        pc_module=pc,
        dataset_root=dataset_root,
        scenes_log=scenes_log,
        scene_source=scene_source,
        episodes=episodes,
        seed=seed,
        cup_variant=variant,
    )
    successes = 0
    # 有界面时默认按仿真时钟播放，便于观看；可用 realtime=False 强制尽快跑完。
    pace_realtime = (render or use_viewer) if realtime is None else realtime
    print("=" * 60)
    print("MuJoCo ACT 闭环评估")
    print("=" * 60)
    print(f"  policy: {policy_path}")
    print(f"  dataset stats: {dataset_root}")
    print(f"  device: {device}")
    print(f"  episodes: {episodes}")
    print(f"  cup: {variant}")
    print(f"  scene source: {scene_source_label}")
    print(f"  max steps: {max_steps} (sim dt 0.002s 时约 {max_steps * 0.002:.1f}s)")
    print(f"  render(head_camera): {render}")
    print(f"  viewer(3D): {use_viewer}")
    print(f"  realtime: {pace_realtime}")
    print(f"  成功: {SUCCESS_CRITERIA_DESCRIPTION}")
    print("=" * 60)

    if use_viewer:
        import mujoco.viewer

    try:
        for index, (scene, scene_label) in enumerate(eval_scenes):
            ctx = SimContext.from_scene(scene)
            renderer = mujoco.Renderer(ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
            t0 = time.time()

            def _run_one(viewer_handle=None):
                return run_policy_episode(
                    ctx,
                    renderer,
                    policy,
                    metadata,
                    preprocess,
                    postprocess,
                    build_inference_frame,
                    make_robot_action,
                    device,
                    max_steps=max_steps,
                    control_hz=RECORD_FPS,
                    action_names=pc.ACTION_NAMES,
                    proprio_state_names=pc.PROPRIO_STATE_NAMES,
                    default_task=DEFAULT_TASK,
                    render=render,
                    viewer=viewer_handle,
                    realtime=pace_realtime,
                )

            if use_viewer:
                with mujoco.viewer.launch_passive(
                    ctx.model,
                    ctx.data,
                    show_right_ui=False,
                ) as viewer:
                    success, metrics = _run_one(viewer)
            else:
                success, metrics = _run_one(None)

            elapsed = time.time() - t0
            if success:
                successes += 1
            status = "OK" if success else "FAIL"
            print(
                f"[{index + 1:3d}/{episodes}] {status} lift={metrics.max_cup_lift:.3f}m "
                f"hold_min={metrics.hold_min_lift:.3f}m "
                f"hold_grasp={metrics.hold_valid_grasp_time:.2f}s "
                f"valid_close={metrics.valid_grasp_at_close} t={elapsed:.1f}s "
                f"{scene_label}"
            )
    finally:
        if render:
            _close_camera_window()

    rate = successes / max(episodes, 1)
    print(f"\n成功率: {successes}/{episodes} = {rate * 100:.1f}%")
    return rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo 闭环评估 ACT 抓杯策略")
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="ACT checkpoint 目录（含 config + weights）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="训练 dataset 根目录（用于 stats/features）",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="每 episode 最大 MuJoCo 仿真步数；3000 步约 6s，接近训练 demo 平均时长",
    )
    parser.add_argument(
        "--scene-source",
        choices=["auto", "success-log", "random"],
        default="auto",
        help=(
            "评估场景来源：auto=优先使用训练日志中的成功场景，找不到则随机；"
            "success-log=必须使用成功日志；random=原始随机场景"
        ),
    )
    parser.add_argument(
        "--scenes-log",
        type=Path,
        default=None,
        help="batch record 日志路径；默认自动查找 picktask/<dataset_name>.log",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="实时弹出 OpenCV 窗口显示 head_camera（策略所见）",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="打开 MuJoCo 3D Viewer；macOS 请用 mjpython 启动本脚本",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按仿真时钟播放（--render/--viewer 时默认开启）",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="即使开了界面也尽快跑完（不等待仿真时钟）",
    )
    parser.add_argument(
        "--cup",
        choices=("yellow", "red"),
        default=None,
        help="杯子变体（默认 yellow；也可用环境变量 PICKCUP_CUP_VARIANT）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_realtime:
        realtime: bool | None = False
    elif args.realtime:
        realtime = True
    else:
        realtime = None
    rate = evaluate(
        args.policy.resolve(),
        args.dataset.resolve(),
        episodes=args.episodes,
        seed=args.seed,
        device_name=args.device,
        max_steps=args.max_steps,
        scene_source=args.scene_source,
        scenes_log=args.scenes_log.resolve() if args.scenes_log else None,
        render=args.render,
        use_viewer=args.viewer,
        realtime=realtime,
        cup_variant=args.cup,
    )
    raise SystemExit(0 if rate > 0.0 else 1)


if __name__ == "__main__":
    main()
