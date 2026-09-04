#!/usr/bin/env python3
"""在 picktask 桌前对杯场景中回放 qk_data/Pouring（LeRobot v3, 6D）。

本脚本位于 qk_validate/，只读复用 picktask 仿真，不改 picktask 源码。

数据关节顺序（Pouring）:
  kRightShoulderPitch, kRightShoulderRoll, kRightShoulderYaw,
  kRightElbow, kRightWristRoll, kWaistYaw

仿真 KEY_CONTROL_JOINTS 顺序:
  waist_yaw, right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll

用法（macOS 需 mjpython 才能开 3D viewer）:
  mjpython qk_validate/replay_pouring.py --episode 0 --loop
  python qk_validate/replay_pouring.py --headless --max-frames 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from _bootstrap import REPO_ROOT, bootstrap

bootstrap()

import mujoco
import mujoco.viewer
import pickcup as pc
from pickcup_sim_core import SimContext

DEFAULT_DATASET = REPO_ROOT / "qk_data" / "Pouring"

# Pouring 6D -> picktask KEY_CONTROL_JOINTS（不含夹爪）
POUR_TO_KEY = [
    ("waist_yaw_joint", 5),
    ("right_shoulder_pitch_joint", 0),
    ("right_shoulder_roll_joint", 1),
    ("right_shoulder_yaw_joint", 2),
    ("right_elbow_joint", 3),
    ("right_wrist_roll_joint", 4),
]


def load_info(dataset: Path) -> dict:
    with (dataset / "meta" / "info.json").open() as f:
        return json.load(f)


def load_episode_table(
    dataset: Path,
    episode: int,
) -> tuple[np.ndarray, np.ndarray, float, Path | None, float]:
    """返回 (states, actions, fps, video_path, video_from_ts)。"""
    info = load_info(dataset)
    fps = float(info.get("fps", 15))
    data_path_tmpl = info["data_path"]
    video_path_tmpl = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    episodes_meta = pq.read_table(
        dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    ep_df = episodes_meta.to_pandas()
    row = ep_df[ep_df["episode_index"] == episode]
    if row.empty:
        raise SystemExit(f"找不到 episode={episode}（数据集共 {len(ep_df)} 条）")
    row = row.iloc[0]
    chunk_index = int(row["data/chunk_index"])
    file_index = int(row["data/file_index"])
    from_idx = int(row["dataset_from_index"])
    to_idx = int(row["dataset_to_index"])

    parquet_path = dataset / data_path_tmpl.format(
        chunk_index=chunk_index,
        file_index=file_index,
    )
    table = pq.read_table(
        parquet_path,
        columns=["index", "episode_index", "frame_index", "observation.state", "action"],
    )
    df = table.to_pandas()
    ep = df[(df["index"] >= from_idx) & (df["index"] < to_idx)].sort_values("frame_index")
    if ep.empty:
        ep = df[df["episode_index"] == episode].sort_values("frame_index")
    if ep.empty:
        raise SystemExit(f"parquet 中无 episode={episode} 帧: {parquet_path}")

    states = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep["action"].to_numpy()).astype(np.float64)

    video_path: Path | None = None
    video_from_ts = 0.0
    v_chunk = int(row.get("videos/observation.images.cam_high/chunk_index", chunk_index))
    v_file = int(row.get("videos/observation.images.cam_high/file_index", file_index))
    video_from_ts = float(row.get("videos/observation.images.cam_high/from_timestamp", 0.0))
    candidate = dataset / video_path_tmpl.format(
        video_key="observation.images.cam_high",
        chunk_index=v_chunk,
        file_index=v_file,
    )
    if candidate.is_file():
        video_path = candidate
    return states, actions, fps, video_path, video_from_ts


def load_dataset_cam_frames(
    video_path: Path | None,
    *,
    from_ts: float,
    n_frames: int,
    fps: float,
) -> list[np.ndarray] | None:
    """从 cam_high mp4 抽取与 episode 对齐的 RGB 帧（失败则返回 None）。"""
    if video_path is None:
        return None
    try:
        import cv2
    except ImportError:
        print(">>> 无 opencv，跳过数据集 cam_high 视频")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f">>> 无法打开数据集视频: {video_path}")
        return None

    start_frame = max(0, int(round(from_ts * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        print(f">>> 数据集视频无帧: {video_path}")
        return None
    print(
        f">>> 已加载数据集 cam_high: {video_path.name} "
        f"frames={len(frames)} from_ts={from_ts:.3f}s"
    )
    return frames


def compose_robot_view(
    sim_rgb: np.ndarray,
    data_rgb: np.ndarray | None,
) -> np.ndarray:
    """拼接：左=仿真 head_camera，右=数据集 cam_high。"""
    left = np.asarray(sim_rgb)
    if data_rgb is None:
        return left
    right = np.asarray(data_rgb)
    if right.shape[:2] != left.shape[:2]:
        try:
            import cv2

            right = cv2.resize(
                right,
                (left.shape[1], left.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            return left
    return np.concatenate([left, right], axis=1)


def pour6_to_ctrl(vec6: np.ndarray, robot_ctrl: dict[str, float]) -> None:
    """把 Pouring 6D 写入 robot_ctrl（夹爪保持张开）。"""
    for joint_name, src_i in POUR_TO_KEY:
        robot_ctrl[joint_name] = float(vec6[src_i])
    robot_ctrl.update(pc.GRIPPER_OPEN)


def force_apply_pose(ctx: SimContext, vec6: np.ndarray) -> None:
    """强制写入关节目标（回放不因桌面碰撞拒绝）。"""
    pour6_to_ctrl(vec6, ctx.robot_ctrl)
    pc.apply_kinematic_teleop(ctx.model, ctx.data, ctx.robot_ctrl)


def physics_for_frame(ctx: SimContext, fps: float) -> None:
    dt = float(ctx.model.opt.timestep)
    n_sub = max(1, int(round((1.0 / fps) / dt)))
    for _ in range(n_sub):
        ctx.step()


def print_banner(
    dataset: Path,
    episode: int,
    n_frames: int,
    fps: float,
    source: str,
    *,
    has_data_cam: bool,
) -> None:
    print("=" * 60)
    print("qk_data/Pouring 回放（桌前正对杯子）")
    print("=" * 60)
    print(f"  dataset : {dataset}")
    print(f"  episode : {episode}  frames={n_frames}  fps={fps}  source={source}")
    print(f"  robot   : pos={pc.DEFAULT_ROBOT_POSITION} yaw=+90°（朝桌）")
    print(f"  cup     : pos={pc.DEFAULT_CUP_POSITION}")
    print("  窗口    : MuJoCo 3D viewer + 机器人视角预览")
    print(
        "  视角    : 左=仿真 head_camera"
        + (" | 右=数据集 cam_high" if has_data_cam else "（无数据集视频）")
    )
    print("  控制    : ESC 退出 | SPACE 暂停/继续 | R 重播")
    print("=" * 60)


def run_headless(args: argparse.Namespace) -> int:
    states, actions, fps, _video, _from_ts = load_episode_table(args.dataset, args.episode)
    traj = states if args.source == "state" else actions
    if args.max_frames is not None:
        traj = traj[: args.max_frames]

    scene = pc.SceneLayout.default(cup_variant=args.cup)
    ctx = SimContext.from_scene(scene)
    force_apply_pose(ctx, traj[0])
    physics_for_frame(ctx, fps)

    for i in range(len(traj)):
        force_apply_pose(ctx, traj[i])
        physics_for_frame(ctx, fps)

    print(
        f"headless OK episode={args.episode} frames={len(traj)} "
        f"final_pitch={ctx.robot_ctrl['right_shoulder_pitch_joint']:.3f}"
    )
    return 0


def run_viewer(args: argparse.Namespace) -> int:
    import glfw
    from head_camera_preview import HeadCameraPreviewProcess

    states, actions, fps, video_path, video_from_ts = load_episode_table(
        args.dataset,
        args.episode,
    )
    traj = states if args.source == "state" else actions
    if args.max_frames is not None:
        traj = traj[: args.max_frames]

    data_frames = None if args.no_data_cam else load_dataset_cam_frames(
        video_path,
        from_ts=video_from_ts,
        n_frames=len(traj),
        fps=fps,
    )
    has_data_cam = data_frames is not None
    preview_w = pc.CAMERA_WIDTH * 2 if has_data_cam else pc.CAMERA_WIDTH
    preview_h = pc.CAMERA_HEIGHT

    scene = pc.SceneLayout.default(cup_variant=args.cup)
    ctx = SimContext.from_scene(scene)
    renderer = mujoco.Renderer(ctx.model, pc.CAMERA_HEIGHT, pc.CAMERA_WIDTH)

    frame_i = 0
    paused = False
    should_exit = False
    next_frame_time = 0.0

    def key_callback(key: int | str) -> None:
        nonlocal should_exit, paused, frame_i, next_frame_time
        key = pc.resolve_key(key)
        if key == glfw.KEY_ESCAPE:
            should_exit = True
        elif key == glfw.KEY_SPACE:
            paused = not paused
            print(">>> 暂停" if paused else ">>> 继续")
        elif key in (glfw.KEY_R, ord("r"), ord("R")):
            frame_i = 0
            ctx.reset()
            force_apply_pose(ctx, traj[0])
            next_frame_time = time.time()
            print(">>> 重播")

    print_banner(
        args.dataset,
        args.episode,
        len(traj),
        fps,
        args.source,
        has_data_cam=has_data_cam,
    )
    force_apply_pose(ctx, traj[0])

    preview = HeadCameraPreviewProcess(
        preview_w,
        preview_h,
        "Pouring robot view: sim | dataset",
        overlay_lines=(
            "LEFT=sim head_camera  RIGHT=dataset cam_high"
            if has_data_cam
            else "sim head_camera",
            f"episode={args.episode}  fps={fps}",
        ),
    )
    preview_ok = [preview.start()]
    if not preview_ok[0]:
        preview.close()
        print(">>> 机器人视角预览启动失败，仍继续打开 3D viewer")

    try:
        with mujoco.viewer.launch_passive(
            ctx.model,
            ctx.data,
            key_callback=key_callback,
            show_right_ui=False,
        ) as viewer:
            pc.set_viewer_robot_camera(viewer, ctx.model, enabled=False)
            next_frame_time = time.time()
            while viewer.is_running() and not should_exit:
                now = time.time()
                preview_rgb: np.ndarray | None = None
                with viewer.lock():
                    if not paused and now >= next_frame_time:
                        force_apply_pose(ctx, traj[frame_i])
                        physics_for_frame(ctx, fps)
                        sim_rgb = pc.render_head_camera(renderer, ctx.data)
                        data_rgb = (
                            data_frames[min(frame_i, len(data_frames) - 1)]
                            if has_data_cam
                            else None
                        )
                        preview_rgb = compose_robot_view(sim_rgb, data_rgb)
                        frame_i += 1
                        next_frame_time = now + (1.0 / fps) / max(args.speed, 1e-6)
                        if frame_i >= len(traj):
                            if args.loop:
                                frame_i = 0
                                ctx.reset()
                                force_apply_pose(ctx, traj[0])
                                print(">>> loop 重播")
                            else:
                                print(
                                    f">>> episode {args.episode} 播放完毕"
                                    "（按 R 重播 / ESC 退出）"
                                )
                                paused = True
                                frame_i = len(traj) - 1
                    elif paused or preview_ok[0]:
                        sim_rgb = pc.render_head_camera(renderer, ctx.data)
                        data_rgb = (
                            data_frames[min(frame_i, len(data_frames) - 1)]
                            if has_data_cam
                            else None
                        )
                        preview_rgb = compose_robot_view(sim_rgb, data_rgb)
                    pc._configure_viewer(viewer)
                viewer.sync()
                if preview_ok[0] and preview_rgb is not None:
                    if not preview.publish(preview_rgb):
                        preview_ok[0] = False
                        print(">>> 机器人视角预览已关闭")
                time.sleep(0.001)
    finally:
        preview.close()
        renderer.close()

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="qk_validate: 回放 qk_data/Pouring 到 picktask 桌前对杯场景"
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="LeRobot v3 数据集根目录",
    )
    p.add_argument("--episode", type=int, default=0, help="episode 编号（0-based）")
    p.add_argument(
        "--source",
        choices=("state", "action"),
        default="state",
        help="回放 observation.state（姿态）或 action（目标）",
    )
    p.add_argument("--speed", type=float, default=1.0, help="播放倍速")
    p.add_argument("--loop", action="store_true", help="循环播放")
    p.add_argument("--max-frames", type=int, default=None, help="最多播放帧数（调试）")
    p.add_argument("--headless", action="store_true", help="无窗口冒烟")
    p.add_argument(
        "--no-data-cam",
        action="store_true",
        help="不加载数据集 cam_high，只显示仿真 head_camera",
    )
    pc.add_cup_cli_argument(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    if not (args.dataset / "meta" / "info.json").is_file():
        raise SystemExit(f"不是有效 LeRobot 数据集: {args.dataset}")
    if args.headless:
        raise SystemExit(run_headless(args))
    raise SystemExit(run_viewer(args))


if __name__ == "__main__":
    main()
