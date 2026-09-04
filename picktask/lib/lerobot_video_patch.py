"""LeRobot 视频解码补丁：torchvision>=0.20 无 VideoReader 时用 PyAV 直接解码。"""

from __future__ import annotations

from pathlib import Path

import av
import torch


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """与 lerobot.datasets.video_utils.decode_video_frames_torchvision 输出一致。"""
    del backend
    video_path = str(video_path)
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    time_base = float(stream.time_base)

    first_ts = min(timestamps)
    last_ts = max(timestamps)

    if first_ts > 0.0:
        seek_target = int(max(0.0, first_ts - 0.05) / time_base)
        container.seek(seek_target, stream=stream)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    for frame in container.decode(video=0):
        ts = float(frame.pts * time_base) if frame.pts is not None else 0.0
        if log_loaded_timestamps:
            print(f"frame loaded at timestamp={ts:.4f}")
        rgb = frame.to_ndarray(format="rgb24")
        loaded_frames.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        loaded_ts.append(ts)
        if ts >= last_ts:
            break

    container.close()

    if not loaded_frames:
        raise RuntimeError(f"未能从视频解码任何帧: {video_path}")

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_dist, argmin = dist.min(1)

    if not (min_dist < tolerance_s).all():
        bad = min_dist[~(min_dist < tolerance_s)]
        raise RuntimeError(
            f"视频帧时间戳偏差过大 ({bad.max().item():.4f}s > {tolerance_s}s): {video_path}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin])
    return closest_frames.type(torch.float32) / 255.0


def apply_lerobot_video_patch() -> None:
    """在 import LeRobotDataset 之前调用。"""
    try:
        import torchvision.io
    except ImportError:
        pass
    else:
        if hasattr(torchvision.io, "VideoReader"):
            return

    import lerobot.datasets.video_utils as video_utils

    video_utils.decode_video_frames_torchvision = decode_video_frames_pyav
