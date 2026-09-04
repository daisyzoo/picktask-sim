"""Simulation context and gripper-free success metrics for liftbag evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import numpy as np

import liftbag as lb
import pickcup as pc

LIFT_HEIGHT_MIN = 0.12
HOLD_TIME_S = 1.5
SUPPORT_CONTACT_GRACE_S = 0.10
THREAD_PLANE_HALF_Y = 0.010

BAG_X_RANGE = (-0.015, 0.015)
BAG_Y_RANGE = (-0.012, 0.012)
BAG_YAW_RANGE = (-0.10, 0.10)
ROBOT_X_RANGE = (-0.03, 0.03)
ROBOT_Y_RANGE = (-0.03, 0.03)


@dataclass(frozen=True)
class LiftBagSceneLayout:
    robot_position: tuple[float, float, float]
    bag_position: tuple[float, float, float]
    bag_yaw: float

    @classmethod
    def default(cls) -> "LiftBagSceneLayout":
        return cls(
            robot_position=tuple(lb.LIFTBAG_ROBOT_POSITION),
            bag_position=tuple(float(value) for value in lb._default_bag_position()),
            bag_yaw=0.0,
        )


def sample_random_scene(rng: np.random.Generator) -> LiftBagSceneLayout:
    default = LiftBagSceneLayout.default()
    robot = np.asarray(default.robot_position, dtype=np.float64)
    bag = np.asarray(default.bag_position, dtype=np.float64)
    robot[0] += rng.uniform(*ROBOT_X_RANGE)
    robot[1] += rng.uniform(*ROBOT_Y_RANGE)
    bag[0] += rng.uniform(*BAG_X_RANGE)
    bag[1] += rng.uniform(*BAG_Y_RANGE)
    return LiftBagSceneLayout(
        robot_position=tuple(float(value) for value in robot),
        bag_position=tuple(float(value) for value in bag),
        bag_yaw=float(rng.uniform(*BAG_YAW_RANGE)),
    )


@dataclass
class LiftBagMetrics:
    crossed_front: bool = False
    crossed_back: bool = False
    threaded_once: bool = False
    supported_once: bool = False
    support_contact_time: float = 0.0
    max_support_contact_time: float = 0.0
    contact_grace: float = 0.0
    max_lift: float = 0.0
    hold_time: float = 0.0
    max_hold_time: float = 0.0
    max_tilt_rad: float = 0.0
    success: bool = False


def _body_collision_geoms(
    model: mujoco.MjModel,
    body_names: tuple[str, ...],
) -> frozenset[int]:
    body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    }
    body_ids.discard(-1)
    return frozenset(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
        and int(model.geom_contype[geom_id]) != 0
        and int(model.geom_conaffinity[geom_id]) != 0
    )


def _named_geoms(model: mujoco.MjModel, names: set[str] | frozenset[str]) -> frozenset[int]:
    result = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in names
    }
    result.discard(-1)
    return frozenset(result)


def _sets_touch(
    data: mujoco.MjData,
    first: frozenset[int],
    second: frozenset[int],
) -> bool:
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if (
            contact.geom1 in first
            and contact.geom2 in second
            or contact.geom2 in first
            and contact.geom1 in second
        ):
            return True
    return False


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
        previous = current
    return inside


def _crosses_aperture_plane(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    plane_y: float,
    aperture_xz: np.ndarray,
) -> bool:
    delta_y = float(current[1] - previous[1])
    if abs(delta_y) < 1e-9:
        return False
    interpolation = (plane_y - float(previous[1])) / delta_y
    if interpolation < 0.0 or interpolation > 1.0:
        return False
    intersection = previous + interpolation * (current - previous)
    return _point_in_polygon(intersection[[0, 2]], aperture_xz)


class LiftBagSimContext:
    def __init__(self, scene: LiftBagSceneLayout) -> None:
        self.scene = scene
        self.model, self.data = lb.make_model_data()
        self.robot_ctrl = pc.INITIAL_QPOS.copy()
        self.actuator_indices = pc.get_actuator_indices(self.model)
        self.teleop_geoms, self.obstacle_geoms = lb.build_liftbag_collision_geom_sets(
            self.model
        )
        self.gripper_geoms = pc.build_named_geom_set(self.model, pc.GRIPPER_GEOM_NAMES)
        arch_names = {
            name for name in lb.BAG_HANDLE_GEOM_NAMES if "_arch_" in name
        }
        self.handle_arch_geoms = _named_geoms(self.model, arch_names)
        self.arm_support_geoms = _body_collision_geoms(
            self.model,
            ("right_elbow_link", "right_wrist_roll_rubber_hand"),
        )
        self.bag_body_id = pc.get_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bag")
        bag_collision_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == self.bag_body_id
            and int(self.model.geom_contype[geom_id]) != 0
        }
        self.obstacle_geoms = frozenset(
            set(self.obstacle_geoms) | bag_collision_geoms
        )
        self.wrist_body_id = pc.get_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_wrist_roll_rubber_hand",
        )
        centerline = lb._handle_centerline(-1.0)
        self.aperture_xz = centerline[:, [0, 2]].copy()
        self.metrics = LiftBagMetrics()
        self.initial_bag_pos = np.zeros(3)
        self.initial_bag_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.previous_wrist_local: np.ndarray | None = None
        self.reset()

    def _world_to_bag_local(self, world_point: np.ndarray) -> np.ndarray:
        bag_pos = self.data.xpos[self.bag_body_id]
        bag_mat = self.data.xmat[self.bag_body_id].reshape(3, 3)
        return bag_mat.T @ (world_point - bag_pos)

    def reset(self) -> None:
        lb.set_fixed_robot_pose(self.model, self.scene.robot_position)
        pc.reset_scene(
            self.model,
            self.data,
            self.robot_ctrl,
            self.actuator_indices,
        )
        lb.set_bag_pose(
            self.model,
            self.data,
            np.asarray(self.scene.bag_position, dtype=np.float64),
            pc.yaw_to_quat(self.scene.bag_yaw),
        )
        mujoco.mj_forward(self.model, self.data)
        self.metrics = LiftBagMetrics()
        self.initial_bag_pos, self.initial_bag_quat = lb.get_bag_pose(
            self.model,
            self.data,
        )
        wrist_world = self.data.xpos[self.wrist_body_id].copy()
        self.previous_wrist_local = self._world_to_bag_local(wrist_world)

    def step(self) -> None:
        pc.apply_hold_ctrl(
            self.model,
            self.data,
            self.robot_ctrl,
            self.actuator_indices,
        )
        pc.apply_kinematic_teleop(self.model, self.data, self.robot_ctrl)
        mujoco.mj_step(self.model, self.data)
        pc.apply_kinematic_teleop(self.model, self.data, self.robot_ctrl)
        pc.sanitize_cup_physics(self.model, self.data, grasp_active=False)
        lb.sanitize_bag_physics(self.model, self.data, grasp_active=False)
        mujoco.mj_forward(self.model, self.data)

    def proprio(self) -> np.ndarray:
        return pc.collect_proprio_state(self.model, self.data)

    def render_rgb(self, renderer: mujoco.Renderer) -> np.ndarray:
        return pc.render_head_camera(renderer, self.data)

    def apply_action_vector(self, action: np.ndarray) -> bool:
        for index, name in enumerate(pc.ACTION_NAMES):
            self.robot_ctrl[name] = float(action[index])
        return pc.try_update_teleop_targets(
            self.model,
            self.data,
            self.robot_ctrl,
            {name: self.robot_ctrl[name] for name in pc.TELEOP_JOINTS},
            self.teleop_geoms,
            self.obstacle_geoms,
        )

    def update_metrics(self, *, sim_dt: float) -> None:
        mujoco.mj_collision(self.model, self.data)
        wrist_world = self.data.xpos[self.wrist_body_id].copy()
        wrist_local = self._world_to_bag_local(wrist_world)
        if self.previous_wrist_local is not None:
            crossed_front_now = _crosses_aperture_plane(
                self.previous_wrist_local,
                wrist_local,
                plane_y=-THREAD_PLANE_HALF_Y,
                aperture_xz=self.aperture_xz,
            )
            crossed_back_now = _crosses_aperture_plane(
                self.previous_wrist_local,
                wrist_local,
                plane_y=THREAD_PLANE_HALF_Y,
                aperture_xz=self.aperture_xz,
            )
            self.metrics.crossed_front = (
                self.metrics.crossed_front or crossed_front_now
            )
            # 机器人从包的 front(-Y) 一侧接近；必须先穿前带，再穿后带。
            self.metrics.crossed_back = self.metrics.crossed_back or (
                self.metrics.crossed_front and crossed_back_now
            )
        self.previous_wrist_local = wrist_local
        self.metrics.threaded_once = (
            self.metrics.threaded_once
            or self.metrics.crossed_front
            and self.metrics.crossed_back
        )

        touching = _sets_touch(
            self.data,
            self.handle_arch_geoms,
            self.arm_support_geoms,
        )
        if self.metrics.threaded_once and touching:
            self.metrics.supported_once = True
            self.metrics.support_contact_time += sim_dt
            self.metrics.contact_grace = SUPPORT_CONTACT_GRACE_S
        else:
            self.metrics.support_contact_time = 0.0
            self.metrics.contact_grace = max(0.0, self.metrics.contact_grace - sim_dt)
        self.metrics.max_support_contact_time = max(
            self.metrics.max_support_contact_time,
            self.metrics.support_contact_time,
        )

        bag_pos, bag_quat = lb.get_bag_pose(self.model, self.data)
        lift = float(bag_pos[2] - self.initial_bag_pos[2])
        self.metrics.max_lift = max(self.metrics.max_lift, lift)
        quat_dot = float(np.clip(abs(np.dot(bag_quat, self.initial_bag_quat)), 0.0, 1.0))
        self.metrics.max_tilt_rad = max(
            self.metrics.max_tilt_rad,
            2.0 * math.acos(quat_dot),
        )

        supported_now = self.metrics.contact_grace > 0.0
        if self.metrics.threaded_once and supported_now and lift >= LIFT_HEIGHT_MIN:
            self.metrics.hold_time += sim_dt
        else:
            self.metrics.hold_time = 0.0
        self.metrics.max_hold_time = max(
            self.metrics.max_hold_time,
            self.metrics.hold_time,
        )
        self.metrics.success = self.metrics.hold_time >= HOLD_TIME_S

    def is_success(self) -> bool:
        return self.metrics.success

    def failure_stage(self) -> str:
        if self.metrics.success:
            return "success"
        if self.metrics.max_lift >= LIFT_HEIGHT_MIN:
            return "lift_unstable"
        if self.metrics.supported_once:
            return "supported_not_lifted"
        if self.metrics.threaded_once:
            return "threaded_not_supported"
        if self.metrics.crossed_front or self.metrics.crossed_back:
            return "partial_thread"
        return "approach"

    def scene_dict(self) -> dict[str, Any]:
        return asdict(self.scene)
