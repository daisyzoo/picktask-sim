"""home_close_cabinet：柜门铰链角接近闭合即成功。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from tasks.base import TaskInfo

CABINET_HINGE_JOINT = "cabinet_stub_hinge"
# 打开角（复位）；闭合成功阈值与保持时间
DEFAULT_OPEN_ANGLE = 0.9  # rad ≈ 51.6°
SUCCESS_ANGLE = 0.08  # |q| < 约 4.6°
HOLD_SECONDS = 0.50


@dataclass
class CloseCabinetTask:
    """关闭西侧 stub 柜门。"""

    info: TaskInfo = field(
        default_factory=lambda: TaskInfo(
            task_id="home_close_cabinet",
            description="关闭柜门",
            success_note=f"|hinge|<{SUCCESS_ANGLE:.2f}rad 保持≥{HOLD_SECONDS:.2f}s",
        )
    )
    open_angle: float = DEFAULT_OPEN_ANGLE
    success_angle: float = SUCCESS_ANGLE
    hold_seconds: float = HOLD_SECONDS
    _joint_id: int = -1
    _qpos_adr: int = -1
    _closed_since: float | None = None

    def bind(self, model: mujoco.MjModel) -> None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CABINET_HINGE_JOINT)
        if jid < 0:
            raise ValueError(f"找不到关节 {CABINET_HINGE_JOINT}")
        self._joint_id = int(jid)
        self._qpos_adr = int(model.jnt_qposadr[jid])

    def hinge_angle(self, data: mujoco.MjData) -> float:
        if self._qpos_adr < 0:
            raise RuntimeError("请先 bind(model)")
        return float(data.qpos[self._qpos_adr])

    def set_hinge_angle(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        angle: float,
    ) -> None:
        if self._qpos_adr < 0:
            self.bind(model)
        low, high = model.jnt_range[self._joint_id]
        data.qpos[self._qpos_adr] = float(np.clip(angle, low, high))
        data.qvel[model.jnt_dofadr[self._joint_id]] = 0.0
        mujoco.mj_forward(model, data)
        self._closed_since = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.bind(model)
        self.set_hinge_angle(model, data, self.open_angle)

    def success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """每仿真步调用；闭合并保持 hold_seconds 后返回 True。"""
        if self._qpos_adr < 0:
            self.bind(model)
        angle = abs(self.hinge_angle(data))
        now = time.monotonic()
        if angle <= self.success_angle:
            if self._closed_since is None:
                self._closed_since = now
            return (now - self._closed_since) >= self.hold_seconds
        self._closed_since = None
        return False

    def success_instant(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """仅看角度，不做保持（单元测试用）。"""
        if self._qpos_adr < 0:
            self.bind(model)
        return abs(self.hinge_angle(data)) <= self.success_angle


def degrees(rad: float) -> float:
    return math.degrees(rad)
