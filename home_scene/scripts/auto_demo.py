#!/usr/bin/env python3
"""home_scene 自动演示 + 头相机长视频录制。

一条长视频：厨房 → 黑场字幕转场 → 关柜 → 转场 → 客厅。
头显视角；Demo 模式只平滑驱动腰+双臂，腿 hold，灯光淡变。

macOS:
  mjpython scripts/auto_demo.py --record
  mjpython scripts/auto_demo.py --record --no-preview
  python scripts/auto_demo.py --headless --record --no-realtime
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from appliances import (  # noqa: E402
    LightFader,
    set_cabinet_door,
    set_light_switch,
    set_stove_knob,
    sync_appliances,
)
from g1_mount import ACTUATED_JOINTS, G1_JOINTS, sync_g1_ctrl_to_qpos  # noqa: E402
from head_camera import (  # noqa: E402
    close_head_camera_preview,
    init_head_camera_preview,
    make_head_renderer,
    render_head_camera,
    show_head_camera_preview,
)
from layout import LAYOUT_CHOICES, layout_from_name  # noqa: E402
from paths import SESSIONS_DATA_DIR, assert_decoupled  # noqa: E402
from scene_composer import configure_viewer, make_model_data  # noqa: E402

RECORD_FPS = 30
FADE_SECONDS = 0.30
TITLE_HOLD_SECONDS = 0.35
DEMO_KP = 36.0
CAMERA_EMA = 0.22  # 轻量压高频抖，避免拖影过重

# 演示只动腰 + 双臂；腿保持初始，避免头显被腿 PD 带着抖
DEMO_JOINTS: tuple[str, ...] = (
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
HOLD_JOINTS: tuple[str, ...] = tuple(j for j in G1_JOINTS if j not in DEMO_JOINTS)


@dataclass
class Cue:
    t: float
    kind: str
    args: dict = field(default_factory=dict)


def kitchen_cues() -> list[Cue]:
    """动作与灯/灶重叠，减少空窗；转腰让炉盘进头显。"""
    return [
        Cue(0.2, "announce", {"text": "① 厨房 · 关灯并转腰"}),
        Cue(
            0.3,
            "pose",
            {
                "duration": 1.2,
                "joints": {
                    "waist_yaw_joint": 0.35,
                    "right_shoulder_pitch_joint": -0.15,
                },
            },
        ),
        Cue(0.5, "light", {"on": False}),
        Cue(1.8, "announce", {"text": "② 开灯"}),
        Cue(1.9, "light", {"on": True}),
        Cue(
            2.2,
            "pose",
            {
                "duration": 1.4,
                "joints": {
                    "waist_yaw_joint": 0.55,
                    "right_shoulder_pitch_joint": -0.25,
                    "right_shoulder_roll_joint": -0.15,
                    "right_elbow_joint": 0.35,
                },
            },
        ),
        Cue(2.6, "announce", {"text": "③ 点灶（看向灶台）"}),
        Cue(2.8, "stove", {"index": 0, "on": True}),
        Cue(3.2, "stove", {"index": 1, "on": True}),
        Cue(3.6, "stove", {"index": 2, "on": True}),
        Cue(4.0, "stove", {"index": 3, "on": True}),
        Cue(
            4.2,
            "pose",
            {
                "duration": 1.6,
                "joints": {
                    "waist_yaw_joint": 0.25,
                    "right_shoulder_pitch_joint": -0.4,
                    "right_elbow_joint": 0.55,
                    "right_wrist_roll_joint": 0.2,
                },
            },
        ),
        Cue(
            6.0,
            "pose",
            {
                "duration": 1.5,
                "joints": {
                    "waist_yaw_joint": -0.1,
                    "right_shoulder_pitch_joint": -0.2,
                    "right_elbow_joint": 0.3,
                },
            },
        ),
        Cue(7.2, "announce", {"text": "④ 熄灶复位"}),
        Cue(7.4, "stove", {"index": 0, "on": False}),
        Cue(7.7, "stove", {"index": 1, "on": False}),
        Cue(8.0, "stove", {"index": 2, "on": False}),
        Cue(8.3, "stove", {"index": 3, "on": False}),
        Cue(
            8.4,
            "pose",
            {
                "duration": 1.4,
                "joints": {
                    "waist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_elbow_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                },
            },
        ),
        Cue(9.4, "done", {}),
    ]


def cabinet_cues() -> list[Cue]:
    return [
        Cue(0.2, "announce", {"text": "⑤ 柜门工位 · 关门"}),
        Cue(0.3, "cabinet", {"open": True}),
        Cue(
            0.5,
            "pose",
            {
                "duration": 1.2,
                "joints": {
                    "waist_yaw_joint": 0.15,
                    "right_shoulder_pitch_joint": -0.3,
                    "right_elbow_joint": 0.45,
                },
            },
        ),
        Cue(1.9, "cabinet", {"open": False}),
        Cue(
            2.2,
            "pose",
            {
                "duration": 1.1,
                "joints": {
                    "waist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_elbow_joint": 0.0,
                },
            },
        ),
        Cue(3.5, "done", {}),
    ]


def living_cues() -> list[Cue]:
    return [
        Cue(0.2, "announce", {"text": "⑥ 客厅 · 工位瞬移"}),
        Cue(
            0.35,
            "pose",
            {
                "duration": 1.2,
                "joints": {
                    "waist_yaw_joint": -0.2,
                    "left_shoulder_pitch_joint": -0.25,
                    "left_elbow_joint": 0.4,
                },
            },
        ),
        Cue(1.4, "light", {"on": False}),
        Cue(2.3, "light", {"on": True}),
        Cue(
            2.4,
            "pose",
            {
                "duration": 1.1,
                "joints": {
                    "waist_yaw_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_elbow_joint": 0.0,
                },
            },
        ),
        Cue(3.7, "announce", {"text": "⑦ 演示结束"}),
        Cue(4.2, "done", {}),
    ]


ALL_SCRIPTS: list[tuple[str, str, list[Cue]]] = [
    ("kitchen", "Kitchen · Counter", kitchen_cues()),
    ("close_cabinet", "Kitchen · Cabinet", cabinet_cues()),
    ("living", "Living · Table", living_cues()),
]


def _ease(u: float) -> float:
    """Smoothstep 5 次多项式，起停加速度为 0。"""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def soft_demo_gains(model: mujoco.MjModel) -> None:
    for name in DEMO_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if aid >= 0:
            model.actuator_gainprm[aid, 0] = DEMO_KP
    for name in HOLD_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if aid >= 0:
            model.actuator_gainprm[aid, 0] = DEMO_KP * 1.4


@dataclass
class PoseBlend:
    start: dict[str, float]
    end: dict[str, float]
    t0: float
    duration: float

    def sample(self, t: float) -> dict[str, float] | None:
        if self.duration <= 1e-6:
            return dict(self.end)
        u = _ease((t - self.t0) / self.duration)
        out = {}
        for k in self.end:
            a = self.start.get(k, self.end[k])
            out[k] = a + (self.end[k] - a) * u
        if t >= self.t0 + self.duration:
            return None  # 完成：调用方应固化 end
        return out


@dataclass
class DemoPlayer:
    model: mujoco.MjModel
    data: mujoco.MjData
    cues: list[Cue]
    ctrl: dict[str, float] = field(default_factory=dict)
    act_ids: dict[str, int] = field(default_factory=dict)
    hold_q: dict[str, float] = field(default_factory=dict)
    sim_t: float = 0.0
    cue_i: int = 0
    finished: bool = False
    blend: PoseBlend | None = None
    lights: LightFader = field(default_factory=LightFader)

    def __post_init__(self) -> None:
        soft_demo_gains(self.model)
        for name in ACTUATED_JOINTS:
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos"
            )
            self.act_ids[name] = int(aid)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            q = float(self.data.qpos[self.model.jnt_qposadr[jid]])
            self.ctrl[name] = q
            if name in HOLD_JOINTS:
                self.hold_q[name] = q
        self.lights.snap(on=True)
        set_light_switch(self.model, self.data, on=True)
        self.apply()
        sync_appliances(self.model, self.data, light_level=self.lights.level)

    def elapsed(self) -> float:
        return self.sim_t

    def apply(self) -> None:
        for name, value in self.ctrl.items():
            self.data.ctrl[self.act_ids[name]] = float(value)

    def _current_demo_pose(self) -> dict[str, float]:
        return {j: self.ctrl[j] for j in DEMO_JOINTS}

    def _start_pose(self, joints: dict[str, float], duration: float) -> None:
        start = self._current_demo_pose()
        end = dict(start)
        end.update({k: float(v) for k, v in joints.items() if k in DEMO_JOINTS})
        self.blend = PoseBlend(
            start=start, end=end, t0=self.sim_t, duration=max(0.2, duration)
        )

    def _fire(self, cue: Cue) -> None:
        kind = cue.kind
        if kind == "announce":
            print(f">>> {cue.args['text']}")
        elif kind == "light":
            on = bool(cue.args["on"])
            set_light_switch(self.model, self.data, on=on)
            self.lights.request(on=on)
            print(f"    light → {'ON' if on else 'OFF'} (fade)")
        elif kind == "stove":
            set_stove_knob(
                self.model,
                self.data,
                int(cue.args["index"]),
                on=bool(cue.args["on"]),
            )
            print(
                f"    stove[{cue.args['index']}] → "
                f"{'ON' if cue.args['on'] else 'OFF'}"
            )
        elif kind == "cabinet":
            set_cabinet_door(self.model, self.data, open_=bool(cue.args["open"]))
            print(f"    cabinet → {'OPEN' if cue.args['open'] else 'CLOSED'}")
        elif kind == "pose":
            self._start_pose(
                cue.args.get("joints", {}),
                float(cue.args.get("duration", 1.2)),
            )
        elif kind == "done":
            self.finished = True
        else:
            raise ValueError(f"未知 cue: {kind}")

    def step(self, dt: float) -> None:
        self.sim_t += dt
        while self.cue_i < len(self.cues) and self.cues[self.cue_i].t <= self.sim_t:
            self._fire(self.cues[self.cue_i])
            self.cue_i += 1

        if self.blend is not None:
            sample = self.blend.sample(self.sim_t)
            if sample is None:
                for k, v in self.blend.end.items():
                    self.ctrl[k] = v
                self.blend = None
            else:
                for k, v in sample.items():
                    self.ctrl[k] = v

        for name, q in self.hold_q.items():
            self.ctrl[name] = q
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

        self.lights.step(dt)
        self.apply()
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qacc).all():
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        sync_appliances(self.model, self.data, light_level=self.lights.level)


class Mp4Recorder:
    def __init__(self, out_dir: Path, fps: int = RECORD_FPS) -> None:
        self.out_dir = out_dir
        self.fps = fps
        self.frames = 0
        self.path = out_dir / "head_camera_demo.mp4"
        self._writer = None
        self._mode = "none"
        self._ema: np.ndarray | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

        # 本机 imageio/PyAV 写帧不稳定，优先 opencv；再试 imageio
        try:
            import cv2

            self._cv2 = cv2
            self._mode = "cv2"
            self._fourcc_tried = ("avc1", "H264", "mp4v")
            print(f"录制 → {self.path} (opencv {fps}fps)")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"opencv 不可用 ({exc})，尝试 imageio")

        import imageio.v2 as imageio

        self._writer = imageio.get_writer(self.path, fps=fps, codec="libx264")
        self._mode = "imageio"
        print(f"录制 → {self.path} (imageio H.264 {fps}fps)")

    def _stabilize(self, rgb: np.ndarray) -> np.ndarray:
        frame = np.asarray(rgb, dtype=np.float32)
        if self._ema is None:
            self._ema = frame.copy()
        else:
            a = CAMERA_EMA
            self._ema = a * frame + (1.0 - a) * self._ema
        return np.clip(self._ema, 0, 255).astype(np.uint8)

    def write(self, rgb: np.ndarray, *, stabilize: bool = True) -> None:
        frame = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        if stabilize:
            frame = self._stabilize(frame)
            frame = np.ascontiguousarray(frame)
        else:
            self._ema = frame.astype(np.float32)
        self.frames += 1
        if self._mode == "imageio":
            self._writer.append_data(frame)
        else:
            if self._writer is None:
                h, w = frame.shape[:2]
                self._writer = None
                for code in getattr(self, "_fourcc_tried", ("mp4v",)):
                    fourcc = self._cv2.VideoWriter_fourcc(*code)
                    writer = self._cv2.VideoWriter(
                        str(self.path), fourcc, float(self.fps), (w, h)
                    )
                    if writer.isOpened():
                        self._writer = writer
                        print(f"  opencv fourcc={code}")
                        break
                    writer.release()
                if self._writer is None:
                    raise RuntimeError("无法创建 VideoWriter")
            bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)

    def write_title_card(self, title: str, seconds: float = TITLE_HOLD_SECONDS) -> None:
        """黑场 + 白字字幕，用于工位转场。"""
        w, h = 640, 480
        n = max(1, int(round(seconds * self.fps)))
        fade_n = max(1, int(round(FADE_SECONDS * self.fps)))
        try:
            import cv2

            base = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                base,
                title,
                (40, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                base,
                "home_scene",
                (40, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )
        except Exception:
            base = np.zeros((h, w, 3), dtype=np.uint8)

        # fade in from black
        for i in range(fade_n):
            a = (i + 1) / fade_n
            self.write((base.astype(np.float32) * a).astype(np.uint8), stabilize=False)
        for _ in range(n):
            self.write(base, stabilize=False)
        for i in range(fade_n):
            a = 1.0 - (i + 1) / fade_n
            self.write((base.astype(np.float32) * a).astype(np.uint8), stabilize=False)
        self._ema = None

    def write_fade_from_frame(self, rgb: np.ndarray, *, to_black: bool) -> None:
        fade_n = max(1, int(round(FADE_SECONDS * self.fps)))
        src = np.asarray(rgb, dtype=np.float32)
        for i in range(fade_n):
            u = (i + 1) / fade_n
            if to_black:
                frame = (src * (1.0 - u)).astype(np.uint8)
            else:
                frame = (src * u).astype(np.uint8)
            self.write(frame, stabilize=False)
        if to_black:
            self._ema = None

    def close(self) -> Path | None:
        if self._mode == "imageio" and self._writer is not None:
            self._writer.close()
            return self.path
        if self._mode == "cv2" and self._writer is not None:
            self._writer.release()
            return self.path
        return None


def _require_mjpython_on_macos(*, headless: bool) -> None:
    if headless or platform.system() != "Darwin":
        return
    import mujoco.viewer as viewer_mod

    if getattr(viewer_mod, "_MJPYTHON", None) is not None:
        return
    print(
        "错误: macOS 上带 viewer 的演示必须用 mjpython：\n"
        "  mjpython scripts/auto_demo.py [--record]",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _new_session_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DATA_DIR / f"demo_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scripts_for(only: str | None) -> list[tuple[str, str, list[Cue]]]:
    if only is None:
        return list(ALL_SCRIPTS)
    return [(n, t, c) for n, t, c in ALL_SCRIPTS if n == only]


def _run_segment(
    layout_name: str,
    cues: list[Cue],
    *,
    recorder: Mp4Recorder | None,
    preview: bool,
    realtime: bool,
    viewer=None,
) -> np.ndarray | None:
    """跑完一段；返回最后一帧 RGB（供淡出）。"""
    layout = layout_from_name(layout_name)
    model, data = make_model_data(layout)
    sync_g1_ctrl_to_qpos(model, data)
    renderer = make_head_renderer(model)
    player = DemoPlayer(model, data, cues)
    next_frame = 0.0
    last_rgb: np.ndarray | None = None

    def _one_step() -> None:
        nonlocal next_frame, last_rgb
        player.step(model.opt.timestep)
        need = recorder is not None or preview
        rgb = render_head_camera(renderer, data) if need else None
        if rgb is not None:
            last_rgb = rgb
            if preview:
                show_head_camera_preview(rgb)
            if recorder is not None and player.elapsed() >= next_frame:
                recorder.write(rgb)
                next_frame += 1.0 / RECORD_FPS

    if viewer is None:
        while not player.finished:
            t0 = time.time()
            _one_step()
            if realtime:
                dt = model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        return last_rgb

    while viewer.is_running() and not player.finished:
        t0 = time.time()
        with viewer.lock():
            player.step(model.opt.timestep)
            need = recorder is not None or preview
            rgb = render_head_camera(renderer, data) if need else None
            configure_viewer(viewer)
        if rgb is not None:
            last_rgb = rgb
            if preview:
                show_head_camera_preview(rgb)
            if recorder is not None and player.elapsed() >= next_frame:
                recorder.write(rgb)
                next_frame += 1.0 / RECORD_FPS
        viewer.sync()
        if realtime:
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    return last_rgb


def run_headless(
    scripts: list[tuple[str, str, list[Cue]]],
    *,
    recorder: Mp4Recorder | None,
    realtime: bool,
) -> None:
    last_rgb = None
    for i, (layout_name, title, cues) in enumerate(scripts):
        if recorder is not None:
            if i == 0:
                recorder.write_title_card(title)
            else:
                if last_rgb is not None:
                    recorder.write_fade_from_frame(last_rgb, to_black=True)
                recorder.write_title_card(title)
        print(f"\n── scene: {layout_name} · {title} ──")
        last_rgb = _run_segment(
            layout_name,
            cues,
            recorder=recorder,
            preview=False,
            realtime=realtime,
        )
    if recorder is not None and last_rgb is not None:
        recorder.write_fade_from_frame(last_rgb, to_black=True)


def run_with_viewer(
    scripts: list[tuple[str, str, list[Cue]]],
    *,
    recorder: Mp4Recorder | None,
    preview: bool,
    realtime: bool,
) -> None:
    import glfw
    import mujoco.viewer

    exit_early = False
    last_rgb = None
    for i, (layout_name, title, cues) in enumerate(scripts):
        if exit_early:
            break
        if recorder is not None:
            if i == 0:
                recorder.write_title_card(title)
            else:
                if last_rgb is not None:
                    recorder.write_fade_from_frame(last_rgb, to_black=True)
                recorder.write_title_card(title)

        print(f"\n── scene: {layout_name} · {title} ──")
        layout = layout_from_name(layout_name)
        model, data = make_model_data(layout)
        sync_g1_ctrl_to_qpos(model, data)
        renderer = make_head_renderer(model)
        player = DemoPlayer(model, data, cues)
        should_skip = False

        def key_callback(key: int | str) -> None:
            nonlocal should_skip, exit_early
            from teleop import resolve_key

            code = resolve_key(key)
            if code == glfw.KEY_ESCAPE:
                should_skip = True
                print(">>> ESC：跳过本段")
            if code == glfw.KEY_Q:
                exit_early = True
                should_skip = True
                print(">>> Q：结束演示")

        next_frame = 0.0
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback, show_right_ui=False
        ) as viewer:
            while (
                viewer.is_running()
                and not player.finished
                and not should_skip
                and not exit_early
            ):
                t0 = time.time()
                rgb = None
                with viewer.lock():
                    player.step(model.opt.timestep)
                    if recorder is not None or preview:
                        rgb = render_head_camera(renderer, data)
                    configure_viewer(viewer)
                if rgb is not None:
                    last_rgb = rgb
                    if preview:
                        show_head_camera_preview(rgb)
                    if recorder is not None and player.elapsed() >= next_frame:
                        recorder.write(rgb)
                        next_frame += 1.0 / RECORD_FPS
                viewer.sync()
                if realtime:
                    dt = model.opt.timestep - (time.time() - t0)
                    if dt > 0:
                        time.sleep(dt)

    if recorder is not None and last_rgb is not None and not exit_early:
        recorder.write_fade_from_frame(last_rgb, to_black=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", action="store_true", help="录制 head_camera 长视频")
    p.add_argument("--headless", action="store_true", help="不打开 MuJoCo viewer")
    p.add_argument("--no-preview", action="store_true", help="关闭头显预览窗")
    p.add_argument("--no-realtime", action="store_true", help="尽快跑完")
    p.add_argument("--only", choices=LAYOUT_CHOICES, help="只跑某一个 layout")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    assert_decoupled()
    _require_mjpython_on_macos(headless=args.headless)
    scripts = _scripts_for(args.only)
    if not scripts:
        print("无匹配的 layout", file=sys.stderr)
        return 2

    session = _new_session_dir() if args.record else None
    recorder = Mp4Recorder(session) if session is not None else None
    preview = False
    if not args.headless and not args.no_preview:
        preview = init_head_camera_preview(enable_by_default=True)

    print("=" * 60)
    print("home_scene 自动演示（头显长视频）")
    print("=" * 60)
    print(f"  layouts : {', '.join(n for n, _, _ in scripts)}")
    print(f"  record  : {session if session else '(off)'}  fps={RECORD_FPS}")
    print(f"  control : waist+arms smooth / legs hold / light fade")
    print(f"  preview : {preview}")
    print("  ESC 跳过本段 / Q 结束")
    print("=" * 60)

    try:
        if args.headless:
            run_headless(
                scripts,
                recorder=recorder,
                realtime=not args.no_realtime,
            )
        else:
            run_with_viewer(
                scripts,
                recorder=recorder,
                preview=preview,
                realtime=not args.no_realtime,
            )
    finally:
        close_head_camera_preview()
        out = recorder.close() if recorder is not None else None
        if out is not None and session is not None and recorder is not None:
            print(f"\n录制完成: {out}  frames={recorder.frames}")
            (session / "README.txt").write_text(
                "home_scene auto_demo (smooth long video)\n"
                f"layouts: {[n for n, _, _ in scripts]}\n"
                f"fps: {RECORD_FPS}\n"
                f"frames: {recorder.frames}\n"
                "camera: head_camera + EMA\n"
                "control: waist+arms quintic / legs hold / light fade\n",
                encoding="utf-8",
            )

    print("\n演示结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
