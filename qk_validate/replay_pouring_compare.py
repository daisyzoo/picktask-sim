#!/usr/bin/env python3
"""Pouring 回放对比（增量版）：左右并排 + 半透明叠图。

复用 `replay_pouring.py` 的数据加载与 6D 驱动逻辑，不改 picktask。

布局模式（运行时 1/2/3 切换）:
  side    — 左仿真 head_camera | 右数据集 cam_high（上一版）
  overlay — α·sim + (1-α)·real 半透明叠图
  both    — 三栏：sim | real | overlay（默认）

用法（macOS 需 mjpython）:
  mjpython qk_validate/replay_pouring_compare.py --episode 0 --loop
  mjpython qk_validate/replay_pouring_compare.py --view both --alpha 0.45
  python qk_validate/replay_pouring_compare.py --headless --max-frames 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap

bootstrap()

import mujoco
import mujoco.viewer
import pickcup as pc
from pickcup_sim_core import SimContext

import replay_pouring as base

ViewMode = str  # "side" | "overlay" | "both"
JOINT_LABELS = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "waist_yaw",
]


def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    src = np.asarray(rgb)
    if src.shape[0] == height and src.shape[1] == width:
        return src
    try:
        import cv2

        return cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        # 无 cv2 时退回切片/垫边（少见）
        out = np.zeros((height, width, 3), dtype=np.uint8)
        h = min(height, src.shape[0])
        w = min(width, src.shape[1])
        out[:h, :w] = src[:h, :w]
        return out


def overlay_blend(sim_rgb: np.ndarray, data_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """α·sim + (1-α)·real；alpha=1 全仿真，alpha=0 全真实。"""
    a = float(np.clip(alpha, 0.0, 1.0))
    left = np.asarray(sim_rgb, dtype=np.float32)
    right = resize_rgb(data_rgb, left.shape[1], left.shape[0]).astype(np.float32)
    blended = a * left + (1.0 - a) * right
    return np.clip(blended, 0, 255).astype(np.uint8)


def annotate_panel(rgb: np.ndarray, title: str) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return np.asarray(rgb)
    out = np.ascontiguousarray(rgb.copy())
    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_joint_error_strip(
    width: int,
    height: int,
    err6: np.ndarray,
    *,
    frame_i: int,
    n_frames: int,
) -> np.ndarray:
    """底部误差条：6 维 |q_sim - q_data| 柱状图。"""
    try:
        import cv2
    except ImportError:
        return np.zeros((height, width, 3), dtype=np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    abs_err = np.abs(np.asarray(err6, dtype=np.float64))
    mean_err = float(abs_err.mean())
    cv2.putText(
        canvas,
        f"joint |err| mean={mean_err:.4f} rad  frame={frame_i}/{n_frames - 1}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    n = len(abs_err)
    margin = 8
    plot_top = 28
    plot_h = height - plot_top - 18
    bar_w = max(8, (width - 2 * margin) // max(n, 1) - 6)
    scale = plot_h / max(0.5, float(abs_err.max()) if abs_err.max() > 1e-6 else 0.5)
    for i, e in enumerate(abs_err):
        x0 = margin + i * ((width - 2 * margin) // n)
        h = int(min(plot_h, e * scale))
        y0 = plot_top + plot_h - h
        color = (80, 200, 80) if e < 0.02 else (60, 160, 220) if e < 0.1 else (60, 60, 220)
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, plot_top + plot_h), color, -1)
        cv2.putText(
            canvas,
            JOINT_LABELS[i][:6],
            (x0, height - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    return canvas


def ctrl6_from_robot(robot_ctrl: dict[str, float]) -> np.ndarray:
    """按 Pouring 6D 顺序读出当前仿真关节。"""
    return np.array(
        [
            robot_ctrl["right_shoulder_pitch_joint"],
            robot_ctrl["right_shoulder_roll_joint"],
            robot_ctrl["right_shoulder_yaw_joint"],
            robot_ctrl["right_elbow_joint"],
            robot_ctrl["right_wrist_roll_joint"],
            robot_ctrl["waist_yaw_joint"],
        ],
        dtype=np.float64,
    )


def compose_compare_view(
    sim_rgb: np.ndarray,
    data_rgb: np.ndarray | None,
    *,
    mode: ViewMode,
    alpha: float,
    err6: np.ndarray | None,
    frame_i: int,
    n_frames: int,
    show_error: bool,
) -> tuple[np.ndarray, int, int]:
    """返回 (rgb, width, height)。"""
    h, w = pc.CAMERA_HEIGHT, pc.CAMERA_WIDTH
    sim = annotate_panel(np.asarray(sim_rgb), "SIM head_camera")
    if data_rgb is None:
        panel = annotate_panel(sim, "SIM only (no dataset cam)")
        if show_error and err6 is not None:
            strip = draw_joint_error_strip(w, 72, err6, frame_i=frame_i, n_frames=n_frames)
            panel = np.concatenate([panel, strip], axis=0)
        return panel, panel.shape[1], panel.shape[0]

    real = annotate_panel(resize_rgb(data_rgb, w, h), "DATA cam_high")
    blend = annotate_panel(
        overlay_blend(sim, real, alpha),
        f"OVERLAY a={alpha:.2f} (1=sim)",
    )

    if mode == "side":
        panel = np.concatenate([sim, real], axis=1)
    elif mode == "overlay":
        panel = blend
    else:  # both
        panel = np.concatenate([sim, real, blend], axis=1)

    if show_error and err6 is not None:
        strip = draw_joint_error_strip(
            panel.shape[1],
            72,
            err6,
            frame_i=frame_i,
            n_frames=n_frames,
        )
        panel = np.concatenate([panel, strip], axis=0)
    return panel, panel.shape[1], panel.shape[0]


def preview_size_for(mode: ViewMode, has_data: bool, show_error: bool) -> tuple[int, int]:
    h = pc.CAMERA_HEIGHT + (72 if show_error else 0)
    if not has_data:
        return pc.CAMERA_WIDTH, h
    if mode == "side":
        return pc.CAMERA_WIDTH * 2, h
    if mode == "overlay":
        return pc.CAMERA_WIDTH, h
    return pc.CAMERA_WIDTH * 3, h


def print_banner(
    dataset: Path,
    episode: int,
    n_frames: int,
    fps: float,
    source: str,
    *,
    mode: ViewMode,
    alpha: float,
) -> None:
    print("=" * 60)
    print("qk_validate Pouring 对比回放（side + overlay）")
    print("=" * 60)
    print(f"  dataset : {dataset}")
    print(f"  episode : {episode}  frames={n_frames}  fps={fps}  source={source}")
    print(f"  view    : {mode}  alpha={alpha:.2f}")
    print("  控制    : ESC退出 SPACE暂停 R重播")
    print("           1=side  2=overlay  3=both")
    print("           [ / ] 降低/提高叠图 alpha（1=全仿真）")
    print("=" * 60)


def run_headless(args: argparse.Namespace) -> int:
    states, actions, fps, _video, _from_ts = base.load_episode_table(
        args.dataset, args.episode
    )
    traj = states if args.source == "state" else actions
    if args.max_frames is not None:
        traj = traj[: args.max_frames]

    scene = pc.SceneLayout.default(cup_variant=args.cup)
    ctx = SimContext.from_scene(scene)
    base.force_apply_pose(ctx, traj[0])
    base.physics_for_frame(ctx, fps)

    errs = []
    for i in range(len(traj)):
        base.force_apply_pose(ctx, traj[i])
        base.physics_for_frame(ctx, fps)
        err = ctrl6_from_robot(ctx.robot_ctrl) - states[min(i, len(states) - 1)]
        errs.append(np.abs(err).mean())
        # 冒烟 compose
        sim = np.zeros((pc.CAMERA_HEIGHT, pc.CAMERA_WIDTH, 3), dtype=np.uint8)
        data = sim.copy()
        compose_compare_view(
            sim,
            data,
            mode=args.view,
            alpha=args.alpha,
            err6=err,
            frame_i=i,
            n_frames=len(traj),
            show_error=not args.no_error,
        )

    print(
        f"headless OK episode={args.episode} frames={len(traj)} "
        f"mean_|err|={float(np.mean(errs)):.5f} view={args.view}"
    )
    return 0


def run_viewer(args: argparse.Namespace) -> int:
    import glfw
    from head_camera_preview import HeadCameraPreviewProcess

    states, actions, fps, video_path, video_from_ts = base.load_episode_table(
        args.dataset,
        args.episode,
    )
    traj = states if args.source == "state" else actions
    if args.max_frames is not None:
        traj = traj[: args.max_frames]

    data_frames = None if args.no_data_cam else base.load_dataset_cam_frames(
        video_path,
        from_ts=video_from_ts,
        n_frames=len(traj),
        fps=fps,
    )
    has_data = data_frames is not None
    show_error = not args.no_error
    mode: ViewMode = args.view
    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    scene = pc.SceneLayout.default(cup_variant=args.cup)
    ctx = SimContext.from_scene(scene)
    renderer = mujoco.Renderer(ctx.model, pc.CAMERA_HEIGHT, pc.CAMERA_WIDTH)

    frame_i = 0
    paused = False
    should_exit = False
    next_frame_time = 0.0
    preview_restart = [False]

    def key_callback(key: int | str) -> None:
        nonlocal should_exit, paused, frame_i, next_frame_time, mode, alpha
        key = pc.resolve_key(key)
        if key == glfw.KEY_ESCAPE:
            should_exit = True
        elif key == glfw.KEY_SPACE:
            paused = not paused
            print(">>> 暂停" if paused else ">>> 继续")
        elif key in (glfw.KEY_R, ord("r"), ord("R")):
            frame_i = 0
            ctx.reset()
            base.force_apply_pose(ctx, traj[0])
            next_frame_time = time.time()
            print(">>> 重播")
        elif key in (glfw.KEY_1, ord("1")):
            mode = "side"
            preview_restart[0] = True
            print(">>> view=side")
        elif key in (glfw.KEY_2, ord("2")):
            mode = "overlay"
            preview_restart[0] = True
            print(">>> view=overlay")
        elif key in (glfw.KEY_3, ord("3")):
            mode = "both"
            preview_restart[0] = True
            print(">>> view=both")
        elif key in (glfw.KEY_LEFT_BRACKET, ord("[")):
            alpha = max(0.0, alpha - 0.05)
            print(f">>> alpha={alpha:.2f}")
        elif key in (glfw.KEY_RIGHT_BRACKET, ord("]")):
            alpha = min(1.0, alpha + 0.05)
            print(f">>> alpha={alpha:.2f}")

    print_banner(
        args.dataset,
        args.episode,
        len(traj),
        fps,
        args.source,
        mode=mode,
        alpha=alpha,
    )
    base.force_apply_pose(ctx, traj[0])

    def make_preview(current_mode: ViewMode) -> HeadCameraPreviewProcess:
        pw, ph = preview_size_for(current_mode, has_data, show_error)
        return HeadCameraPreviewProcess(
            pw,
            ph,
            f"Pouring compare [{current_mode}]",
            overlay_lines=(
                f"mode={current_mode}  alpha={alpha:.2f}",
                "1 side | 2 overlay | 3 both | [ ] alpha",
            ),
        )

    preview = make_preview(mode)
    preview_ok = [preview.start()]
    if not preview_ok[0]:
        preview.close()
        print(">>> 预览启动失败，仍继续 3D viewer")

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
                if preview_restart[0]:
                    preview_restart[0] = False
                    preview.close()
                    preview = make_preview(mode)
                    preview_ok[0] = preview.start()
                    if not preview_ok[0]:
                        print(">>> 切换布局后预览重启失败")

                now = time.time()
                preview_rgb: np.ndarray | None = None
                with viewer.lock():
                    if not paused and now >= next_frame_time:
                        base.force_apply_pose(ctx, traj[frame_i])
                        base.physics_for_frame(ctx, fps)
                        sim_rgb = pc.render_head_camera(renderer, ctx.data)
                        data_rgb = (
                            data_frames[min(frame_i, len(data_frames) - 1)]
                            if has_data
                            else None
                        )
                        err6 = ctrl6_from_robot(ctx.robot_ctrl) - states[
                            min(frame_i, len(states) - 1)
                        ]
                        preview_rgb, _, _ = compose_compare_view(
                            sim_rgb,
                            data_rgb,
                            mode=mode,
                            alpha=alpha,
                            err6=err6,
                            frame_i=frame_i,
                            n_frames=len(traj),
                            show_error=show_error,
                        )
                        frame_i += 1
                        next_frame_time = now + (1.0 / fps) / max(args.speed, 1e-6)
                        if frame_i >= len(traj):
                            if args.loop:
                                frame_i = 0
                                ctx.reset()
                                base.force_apply_pose(ctx, traj[0])
                                print(">>> loop 重播")
                            else:
                                print(
                                    f">>> episode {args.episode} 播放完毕"
                                    "（R 重播 / ESC 退出）"
                                )
                                paused = True
                                frame_i = len(traj) - 1
                    elif paused or preview_ok[0]:
                        sim_rgb = pc.render_head_camera(renderer, ctx.data)
                        data_rgb = (
                            data_frames[min(frame_i, len(data_frames) - 1)]
                            if has_data
                            else None
                        )
                        err6 = ctrl6_from_robot(ctx.robot_ctrl) - states[
                            min(frame_i, len(states) - 1)
                        ]
                        preview_rgb, _, _ = compose_compare_view(
                            sim_rgb,
                            data_rgb,
                            mode=mode,
                            alpha=alpha,
                            err6=err6,
                            frame_i=frame_i,
                            n_frames=len(traj),
                            show_error=show_error,
                        )
                    pc._configure_viewer(viewer)
                viewer.sync()
                if preview_ok[0] and preview_rgb is not None:
                    # 切换 mode 后尺寸可能变化：以当前 preview 为准再 resize
                    need = (preview.height, preview.width, 3)
                    if preview_rgb.shape != need:
                        preview_rgb = resize_rgb(
                            preview_rgb, preview.width, preview.height
                        )
                    if not preview.publish(preview_rgb):
                        preview_ok[0] = False
                        print(">>> 预览已关闭")
                time.sleep(0.001)
    finally:
        preview.close()
        renderer.close()

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="qk_validate: Pouring side+overlay 对比回放（增量）"
    )
    p.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--source", choices=("state", "action"), default="state")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-data-cam", action="store_true")
    p.add_argument(
        "--view",
        choices=("side", "overlay", "both"),
        default="both",
        help="预览布局：side / overlay / both（默认 both）",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="叠图系数：1=全仿真，0=全真实",
    )
    p.add_argument(
        "--no-error",
        action="store_true",
        help="不显示底部 6D 关节误差条",
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
