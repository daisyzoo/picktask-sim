"""在独立普通 Python 进程中显示机器人相机，避免 mjpython 与 Qt/Cocoa 冲突。"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Sequence

import numpy as np

_HEADER_SIZE = 16
_STOP_OFFSET = 0
_STATUS_OFFSET = 1
_SEQUENCE_OFFSET = 8

_STATUS_STARTING = 0
_STATUS_READY = 1
_STATUS_ERROR = 2
_STATUS_CLOSED = 3


class HeadCameraPreviewProcess:
    """用共享内存向独立 OpenCV GUI 进程发布最新 RGB 帧。"""

    def __init__(
        self,
        width: int,
        height: int,
        window_name: str,
        *,
        overlay_lines: Sequence[str] = (),
    ) -> None:
        self.width = width
        self.height = height
        self.window_name = window_name
        self.overlay_lines = tuple(overlay_lines)
        self._frame_bytes = width * height * 3
        self._shm: shared_memory.SharedMemory | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence = 0

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, *, timeout: float = 5.0) -> bool:
        if self.is_alive:
            return True
        self.close()

        self._shm = shared_memory.SharedMemory(
            create=True,
            size=_HEADER_SIZE + self._frame_bytes,
        )
        self._shm.buf[:_HEADER_SIZE] = b"\0" * _HEADER_SIZE

        python = Path(sys.prefix) / "bin" / "python"
        if not python.is_file():
            print(f"机器人预览启动失败: 找不到普通 Python: {python}")
            self.close()
            return False

        command = [
            str(python),
            str(Path(__file__).resolve()),
            "--child",
            "--shm-name",
            self._shm.name,
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--window-name",
            self.window_name,
            "--overlay-lines",
            json.dumps(self.overlay_lines, ensure_ascii=False),
        ]
        try:
            self._process = subprocess.Popen(command)
        except OSError as exc:
            print(f"机器人预览启动失败: {exc}")
            self.close()
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = int(self._shm.buf[_STATUS_OFFSET])
            if status == _STATUS_READY:
                return True
            if status in {_STATUS_ERROR, _STATUS_CLOSED} or not self.is_alive:
                self.close()
                return False
            time.sleep(0.02)

        print("机器人预览启动超时")
        self.close()
        return False

    def publish(self, rgb: np.ndarray) -> bool:
        """发布最新 RGB；子进程退出时返回 False。"""
        if not self.is_alive or self._shm is None:
            return False
        status = int(self._shm.buf[_STATUS_OFFSET])
        if status in {_STATUS_ERROR, _STATUS_CLOSED}:
            return False
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(
                f"机器人预览帧尺寸应为 {(self.height, self.width, 3)}，实际为 {rgb.shape}"
            )

        frame = np.asarray(rgb, dtype=np.uint8)
        target = np.ndarray(
            (self.height, self.width, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=_HEADER_SIZE,
        )

        # 奇数表示正在写，偶数表示一帧已完整提交；子进程据此避免读取撕裂帧。
        self._sequence += 2
        struct.pack_into("<Q", self._shm.buf, _SEQUENCE_OFFSET, self._sequence - 1)
        np.copyto(target, frame)
        struct.pack_into("<Q", self._shm.buf, _SEQUENCE_OFFSET, self._sequence)
        return True

    def close(self) -> None:
        process = self._process
        shm = self._shm
        self._process = None
        self._shm = None

        if shm is not None:
            try:
                shm.buf[_STOP_OFFSET] = 1
            except (BufferError, TypeError):
                pass

        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

        if shm is not None:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass


def _annotate_frame(frame: np.ndarray, lines: Sequence[str], cv2) -> np.ndarray:
    annotated = np.ascontiguousarray(frame.copy())
    y = 22
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 220, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return annotated


def _run_child(args: argparse.Namespace) -> int:
    shm = shared_memory.SharedMemory(name=args.shm_name)
    # 父进程拥有并负责 unlink；避免子进程 resource_tracker 抢先删除。
    resource_tracker.unregister(shm._name, "shared_memory")
    overlay_lines = tuple(json.loads(args.overlay_lines))

    try:
        import cv2
    except ImportError as exc:
        shm.buf[_STATUS_OFFSET] = _STATUS_ERROR
        print(f"机器人视角预览已禁用: 未安装 OpenCV: {exc}", flush=True)
        shm.close()
        return 1

    frame_view = np.ndarray(
        (args.height, args.width, 3),
        dtype=np.uint8,
        buffer=shm.buf,
        offset=_HEADER_SIZE,
    )
    last_sequence = 0
    # macOS/OpenCV：窗口刚创建时 WND_PROP_VISIBLE 常返回 -1（不支持/未知），
    # 旧逻辑把 <1 当成已关闭，会导致窗口一闪即灭。
    visibility_grace_until = time.monotonic() + 1.5
    invisible_streak = 0
    try:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, args.width, args.height)
        # 先画一帧占位，避免空窗被系统立刻回收。
        placeholder = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        placeholder[:] = (32, 32, 32)
        cv2.imshow(
            args.window_name,
            _annotate_frame(placeholder, overlay_lines or ("waiting for frames…",), cv2),
        )
        cv2.waitKey(1)
        shm.buf[_STATUS_OFFSET] = _STATUS_READY

        while not shm.buf[_STOP_OFFSET]:
            sequence_before = struct.unpack_from("<Q", shm.buf, _SEQUENCE_OFFSET)[0]
            if sequence_before and sequence_before % 2 == 0 and sequence_before != last_sequence:
                rgb = frame_view.copy()
                sequence_after = struct.unpack_from("<Q", shm.buf, _SEQUENCE_OFFSET)[0]
                if sequence_before == sequence_after:
                    annotated = _annotate_frame(rgb, overlay_lines, cv2)
                    cv2.imshow(args.window_name, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    last_sequence = sequence_after

            key = cv2.waitKey(10) & 0xFF
            if key in {27, ord("q"), ord("Q")}:  # ESC / q 手动关预览
                break

            if time.monotonic() >= visibility_grace_until:
                try:
                    visible = cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    visible = 1.0
                # 仅当明确返回 0（不可见）时认为用户关窗；-1 表示属性不可用，忽略。
                if visible == 0:
                    invisible_streak += 1
                    if invisible_streak >= 5:
                        break
                else:
                    invisible_streak = 0
    except cv2.error as exc:
        shm.buf[_STATUS_OFFSET] = _STATUS_ERROR
        print(f"机器人视角预览窗口异常: {exc}", flush=True)
        return 1
    finally:
        if shm.buf[_STATUS_OFFSET] != _STATUS_ERROR:
            shm.buf[_STATUS_OFFSET] = _STATUS_CLOSED
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        shm.close()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--window-name", required=True)
    parser.add_argument("--overlay-lines", default="[]")
    return parser.parse_args()


if __name__ == "__main__":
    child_args = _parse_args()
    if not child_args.child:
        raise SystemExit("该模块仅供机器人预览子进程调用")
    raise SystemExit(_run_child(child_args))
