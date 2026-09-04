"""LeRobot v3 dataset recorder for pickcup MuJoCo teleop / scripted demos."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from paths import PICKTASK_DIR, setup_lib_paths

setup_lib_paths(include_lerobot=True)

RECORD_FPS = 30
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
HEAD_CAMERA_FEATURE = "observation.images.head_camera"
DEFAULT_TASK = "pick up the cup"
REPO_ID_PREFIX = "local/pickcup"


def _av_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


def build_dataset_features(
    state_names: list[str],
    action_names: list[str],
    *,
    use_video: bool = True,
) -> dict:
    image_dtype = "video" if use_video else "image"
    return {
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        HEAD_CAMERA_FEATURE: {
            "dtype": image_dtype,
            "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }


class PickcupLeRobotRecorder:
    """Manage LeRobot dataset sessions for pickcup."""

    def __init__(
        self,
        data_dir: Path,
        state_names: list[str],
        action_names: list[str],
        *,
        task: str = DEFAULT_TASK,
        robot_type: str = "unitree_g1_23dof",
        session_name: str | None = None,
        session_prefix: str = "pickcup",
    ) -> None:
        self.data_dir = data_dir
        self.state_names = state_names
        self.action_names = action_names
        self.task = task
        self.robot_type = robot_type
        self._session_name = session_name
        self._session_prefix = session_prefix
        self._use_video = _av_available()
        self.features = build_dataset_features(
            state_names,
            action_names,
            use_video=self._use_video,
        )
        self.dataset = None
        self.session_root: Path | None = None
        self._session_index = 0

    @property
    def is_active(self) -> bool:
        return self.dataset is not None

    @staticmethod
    def _import_lerobot_dataset():
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                f"LeRobot 录制依赖未安装。请运行: pip install -r {PICKTASK_DIR / 'requirements-recording.txt'}"
            ) from exc
        return LeRobotDataset

    def _make_session_root(self) -> Path:
        if self._session_name:
            return self.data_dir / self._session_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.data_dir / f"{self._session_prefix}_{timestamp}"

    def start_session(self) -> Path:
        """Create a new LeRobot dataset directory."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_root = self._make_session_root()
        repo_id = f"{REPO_ID_PREFIX}_{self.session_root.name}"
        create_kwargs = {
            "repo_id": repo_id,
            "fps": RECORD_FPS,
            "features": self.features,
            "root": self.session_root,
            "robot_type": self.robot_type,
            "use_videos": self._use_video,
        }
        if self._use_video:
            create_kwargs.update(
                {
                    "vcodec": "h264",
                    "streaming_encoding": True,
                    "encoder_threads": 2,
                }
            )
        else:
            print(
                "警告: 未检测到 PyAV (av)，head_camera 将以 image 格式写入。"
                "安装 av 后可导出 mp4: conda install -c conda-forge av"
            )
        LeRobotDataset = self._import_lerobot_dataset()
        self.dataset = LeRobotDataset.create(**create_kwargs)
        self._session_index += 1
        print(f"\nLeRobot session: {self.session_root}")
        return self.session_root

    def discard_episode(self) -> None:
        """Drop buffered frames for a failed episode."""
        if self.dataset is None:
            return
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def save_episode(self) -> None:
        if self.dataset is None:
            raise RuntimeError("LeRobot recorder session is not active")
        if not self.dataset.has_pending_frames():
            return
        self.dataset.save_episode()
        print(f"  - episode saved ({self.dataset.meta.total_episodes} total)")

    def close_session(self) -> None:
        """Save pending episode and finalize the current dataset."""
        if self.dataset is None:
            return
        if self.dataset.has_pending_frames():
            self.dataset.save_episode()
            print(f"  - episode saved ({self.dataset.meta.total_episodes} total)")
        self.dataset.finalize()
        print(f"  - dataset finalized: {self.session_root}")
        self.dataset = None

    def restart_session(self) -> Path:
        self.close_session()
        return self.start_session()

    def add_frame(
        self,
        *,
        observation_state: np.ndarray,
        action: np.ndarray,
        rgb: np.ndarray,
        task: str | None = None,
    ) -> None:
        if self.dataset is None:
            raise RuntimeError("LeRobot recorder session is not active")

        frame = {
            "action": action.astype(np.float32, copy=False),
            "observation.state": observation_state.astype(np.float32, copy=False),
            HEAD_CAMERA_FEATURE: np.ascontiguousarray(rgb, dtype=np.uint8),
            "task": task or self.task,
        }
        self.dataset.add_frame(frame)
