"""Shared MuJoCo pick-cup simulation logic for scripted demos, batch recording, and policy eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np

import pickcup as pc

# 成功标准：见 SUCCESS_CRITERIA_DESCRIPTION / check_episode_success。
LIFT_HEIGHT_MIN = 0.05
HOLD_TIME = 1.5
SETTLE_TIME = 0.8
APPROACH_STEP_PERIOD = 0.04
PRE_CLOSE_DOWN_STEPS = 0
ALIGN_STEP_PERIOD = 0.10
ALIGN_MAX_STEPS = 12
CLOSE_WAIT_TIME = 1.5
LIFT_STEP_PERIOD = 0.08
MAX_LIFT_STEPS = 80

DEFAULT_APPROACH_TUNING = {
    "right_shoulder_pitch_joint": 0.30,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.50,
}

# (forward_steps, down_steps, 末端关节微调)；离散 REACH 步进避免插值轨迹推走杯子。
APPROACH_TRAJECTORY_VARIANTS = [
    (12, 4, dict(DEFAULT_APPROACH_TUNING)),
    (11, 4, dict(DEFAULT_APPROACH_TUNING)),
    (12, 5, dict(DEFAULT_APPROACH_TUNING)),
    (13, 4, dict(DEFAULT_APPROACH_TUNING)),
]

# 兼容旧引用
APPROACH_POSE_VARIANTS = APPROACH_TRAJECTORY_VARIANTS

SUCCESS_CRITERIA_DESCRIPTION = (
    "右臂手心朝杯闭合+有效指间接触, 抬起>5cm, "
    f"并在 {HOLD_TIME:.1f}s 内保持合理抓取姿态"
)

_ZERO_RIGHT_ARM = {
    "right_shoulder_pitch_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
}


class DemoPhase(Enum):
    SETTLE = auto()
    APPROACH = auto()
    ALIGN = auto()
    PRE_CLOSE = auto()
    CLOSE = auto()
    LIFT = auto()
    HOLD = auto()
    DONE = auto()


@dataclass
class EpisodeMetrics:
    initial_cup_z: float = 0.0
    max_cup_lift: float = 0.0
    hold_min_lift: float = 0.0
    hold_valid_grasp_time: float = 0.0
    valid_grasp_at_close: bool = False
    had_contact_at_close: bool = False
    grasp_active: bool = False

    @property
    def lift_achieved(self) -> bool:
        return self.max_cup_lift >= LIFT_HEIGHT_MIN

    @property
    def success(self) -> bool:
        return (
            self.grasp_active
            and self.valid_grasp_at_close
            and self.hold_min_lift >= LIFT_HEIGHT_MIN
            and self.hold_valid_grasp_time + 1e-6 >= HOLD_TIME
        )


@dataclass
class AutoPickController:
    """Scripted 抓杯状态机；朝 +Y 站位时抬起用 REACH_UP。"""

    phase: DemoPhase = DemoPhase.SETTLE
    phase_elapsed: float = 0.0
    approach_waypoints: list[dict[str, float]] = field(default_factory=list)
    approach_total_steps: int = 0
    approach_index: int = 0
    pre_close_index: int = 0
    align_index: int = 0
    lift_index: int = 0
    close_applied: bool = False
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)
    variant_index: int = 0

    def reset(
        self,
        start_arm: dict[str, float],
        scene: pc.SceneLayout,
        *,
        variant_index: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.phase = DemoPhase.SETTLE
        self.phase_elapsed = 0.0
        self.approach_index = 0
        self.pre_close_index = 0
        self.align_index = 0
        self.lift_index = 0
        self.close_applied = False
        self.metrics = EpisodeMetrics()
        self.variant_index = variant_index
        forward_steps, down_steps, tuning = plan_approach_trajectory(
            scene,
            variant_index=self.variant_index,
            rng=rng,
        )
        self.approach_waypoints = build_delta_approach_waypoints(
            start_arm,
            forward_steps,
            down_steps,
            tuning,
        )
        self.approach_total_steps = len(self.approach_waypoints)

    @property
    def finished(self) -> bool:
        return self.phase == DemoPhase.DONE

    def bind_initial_cup_z(self, cup_z: float) -> None:
        self.metrics.initial_cup_z = cup_z
        self.metrics.max_cup_lift = 0.0
        self.metrics.hold_min_lift = 0.0
        self.metrics.hold_valid_grasp_time = 0.0
        self.metrics.valid_grasp_at_close = False
        self.metrics.had_contact_at_close = False

    def update_cup_tracking(self, cup_z: float) -> None:
        lift = float(cup_z - self.metrics.initial_cup_z)
        self.metrics.max_cup_lift = max(self.metrics.max_cup_lift, lift)
        if self.phase == DemoPhase.HOLD:
            if self.metrics.hold_min_lift == 0.0 and lift > 0.0:
                self.metrics.hold_min_lift = lift
            elif self.metrics.hold_min_lift > 0.0:
                self.metrics.hold_min_lift = min(self.metrics.hold_min_lift, lift)

    def advance(
        self,
        sim_dt: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_ctrl: dict[str, float],
        grasp: pc.GraspState,
        teleop_geoms: frozenset[int],
        obstacle_geoms: frozenset[int],
        gripper_geoms: frozenset[int],
        cup_geoms: frozenset[int],
    ) -> None:
        cup_z = float(pc.get_cup_pose(model, data)[0][2])
        self.update_cup_tracking(cup_z)
        self.metrics.grasp_active = grasp.active
        self.phase_elapsed += sim_dt

        if self.phase == DemoPhase.SETTLE:
            if self.phase_elapsed >= SETTLE_TIME:
                self._enter(DemoPhase.APPROACH)
            return

        if self.phase == DemoPhase.APPROACH:
            target_index = min(
                self.approach_total_steps,
                int(self.phase_elapsed / APPROACH_STEP_PERIOD),
            )
            while self.approach_index < target_index:
                self.approach_index += 1
                waypoint = self.approach_waypoints[self.approach_index - 1]
                pc.try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    waypoint,
                    teleop_geoms,
                    obstacle_geoms,
                )
            if self.approach_index >= self.approach_total_steps:
                self._enter(DemoPhase.ALIGN)
            return

        if self.phase == DemoPhase.ALIGN:
            step_index = int(self.phase_elapsed / ALIGN_STEP_PERIOD)
            while self.align_index < min(ALIGN_MAX_STEPS, step_index):
                self.align_index += 1
                apply_alignment_correction(
                    model,
                    data,
                    robot_ctrl,
                    teleop_geoms,
                    obstacle_geoms,
                )
            mujoco.mj_collision(model, data)
            cup_pos = pc.get_cup_pose(model, data)[0]
            gripper_center = pc.get_gripper_center(model, data)
            dist = float(np.linalg.norm(cup_pos - gripper_center))
            (
                _,
                _,
                _,
                _,
                approach_dot,
                _dorsal_dot,
                _lateral_dot,
                finger_lat,
                _cup_above,
            ) = pc._grasp_alignment(model, data)
            aligned = (
                approach_dot >= pc.GRASP_MIN_APPROACH_DOT - 0.05
                and finger_lat <= pc.GRASP_MAX_FINGER_LATERAL_OFFSET + 0.008
                and dist <= 0.155
            )
            if aligned or self.align_index >= ALIGN_MAX_STEPS:
                if PRE_CLOSE_DOWN_STEPS > 0:
                    self._enter(DemoPhase.PRE_CLOSE)
                else:
                    self._enter(DemoPhase.CLOSE)
            return

        if self.phase == DemoPhase.PRE_CLOSE:
            step_index = int(self.phase_elapsed / LIFT_STEP_PERIOD)
            while self.pre_close_index < min(PRE_CLOSE_DOWN_STEPS, step_index):
                self.pre_close_index += 1
                apply_arm_deltas(
                    model,
                    data,
                    robot_ctrl,
                    pc.REACH_DOWN_DELTAS,
                    teleop_geoms,
                    obstacle_geoms,
                )
            if self.pre_close_index >= PRE_CLOSE_DOWN_STEPS:
                self._enter(DemoPhase.CLOSE)
            return

        if self.phase == DemoPhase.CLOSE:
            if not self.close_applied:
                robot_ctrl.update(pc.GRIPPER_CLOSED)
                pc.try_update_teleop_targets(
                    model,
                    data,
                    robot_ctrl,
                    pc.GRIPPER_CLOSED,
                    teleop_geoms,
                    obstacle_geoms,
                )
                self.close_applied = True
            mujoco.mj_collision(model, data)
            if pc.can_initiate_grasp(
                model,
                data,
                robot_ctrl,
                gripper_geoms,
                cup_geoms,
            ):
                self.metrics.valid_grasp_at_close = True
                self.metrics.had_contact_at_close = True
            if grasp.active:
                self._enter(DemoPhase.LIFT)
            elif self.phase_elapsed >= CLOSE_WAIT_TIME:
                self._enter(DemoPhase.DONE)
            return

        if self.phase == DemoPhase.LIFT:
            step_index = int(self.phase_elapsed / LIFT_STEP_PERIOD)
            while self.lift_index < min(MAX_LIFT_STEPS, step_index):
                self.lift_index += 1
                if grasp.active and pc.can_maintain_grasp(
                    model,
                    data,
                    robot_ctrl,
                    gripper_geoms,
                    cup_geoms,
                ):
                    apply_arm_deltas(
                        model,
                        data,
                        robot_ctrl,
                        pc.REACH_UP_DELTAS,
                        teleop_geoms,
                        obstacle_geoms,
                    )
            lift = cup_z - self.metrics.initial_cup_z
            lift_ok = lift >= LIFT_HEIGHT_MIN
            if (lift_ok and grasp.active) or self.lift_index >= MAX_LIFT_STEPS:
                if lift_ok and grasp.active:
                    self._enter(DemoPhase.HOLD)
                else:
                    self._enter(DemoPhase.DONE)
            return

        if self.phase == DemoPhase.HOLD:
            lift = cup_z - self.metrics.initial_cup_z
            if (
                grasp.active
                and pc.is_gripper_closed(robot_ctrl)
                and lift >= LIFT_HEIGHT_MIN
                and pc.can_maintain_grasp(
                    model,
                    data,
                    robot_ctrl,
                    gripper_geoms,
                    cup_geoms,
                )
            ):
                self.metrics.hold_valid_grasp_time += sim_dt
            if self.phase_elapsed >= HOLD_TIME:
                self._enter(DemoPhase.DONE)
            return

    def _enter(self, phase: DemoPhase) -> None:
        self.phase = phase
        self.phase_elapsed = 0.0
        if phase == DemoPhase.HOLD:
            lift = self.metrics.max_cup_lift
            self.metrics.hold_min_lift = lift if lift > 0.0 else 0.0


def _scene_reach_distance(scene: pc.SceneLayout) -> float:
    rx, ry = scene.robot_position[0], scene.robot_position[1]
    cx, cy = scene.cup_position[0], scene.cup_position[1]
    return float(np.hypot(cx - rx, cy - ry))


def plan_approach_trajectory(
    scene: pc.SceneLayout,
    variant_index: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, dict[str, float]]:
    """按场景几何规划离散 REACH 步数与末端微调。"""
    default_rx, default_ry = pc.DEFAULT_ROBOT_POSITION[0], pc.DEFAULT_ROBOT_POSITION[1]
    default_cx, default_cy = pc.DEFAULT_CUP_POSITION[0], pc.DEFAULT_CUP_POSITION[1]
    default_reach = float(np.hypot(default_cx - default_rx, default_cy - default_ry))
    reach = _scene_reach_distance(scene)
    reach_delta = reach - default_reach
    # 每步 REACH_FORWARD 约缩短 0.011m 夹爪-杯子水平距离（经验拟合）
    forward_steps = int(np.clip(round(12 + reach_delta / 0.011), 8, 18))
    cup_dy = scene.cup_position[1] - pc.DEFAULT_CUP_POSITION[1]
    down_steps = int(np.clip(round(4 - cup_dy * 3.0), 2, 7))

    tuning = dict(DEFAULT_APPROACH_TUNING)
    variant = variant_index % 4
    forward_steps = int(np.clip(forward_steps + [-2, -1, 0, 1][variant], 8, 18))
    tuning = adapt_approach_tuning(tuning, scene, rng)
    return forward_steps, down_steps, tuning


def adapt_approach_tuning(
    tuning: dict[str, float],
    scene: pc.SceneLayout,
    rng: np.random.Generator | None,
) -> dict[str, float]:
    """根据杯子/机器人位置微调 approach 末端肩关节，并加小扰动（不改写肘角）。"""
    cup_dx = scene.cup_position[0] - pc.DEFAULT_CUP_POSITION[0]
    cup_dy = scene.cup_position[1] - pc.DEFAULT_CUP_POSITION[1]
    robot_dx = scene.robot_position[0] - pc.DEFAULT_ROBOT_POSITION[0]
    robot_dy = scene.robot_position[1] - pc.DEFAULT_ROBOT_POSITION[1]
    pitch = float(tuning["right_shoulder_pitch_joint"] - 0.65 * cup_dx - 0.35 * robot_dx)
    roll = float(
        tuning["right_shoulder_roll_joint"]
        + 0.50 * cup_dy
        + 0.25 * robot_dy
        + 0.55 * cup_dx
    )
    yaw = float(
        tuning["right_shoulder_yaw_joint"]
        - 0.55 * cup_dy
        - 0.30 * robot_dy
        + 0.35 * cup_dx
    )
    elbow_delta = float(0.15 * scene.cup_yaw)
    if rng is not None:
        pitch += float(rng.uniform(-0.04, 0.04))
        roll += float(rng.uniform(-0.03, 0.03))
        yaw += float(rng.uniform(-0.05, 0.05))
        elbow_delta += float(rng.uniform(-0.04, 0.04))
    out = {
        "right_shoulder_pitch_joint": pitch,
        "right_shoulder_roll_joint": roll,
        "right_shoulder_yaw_joint": yaw,
    }
    if abs(elbow_delta) > 1e-9:
        out["right_elbow_delta"] = elbow_delta
    return out


def adapt_approach_pose(
    base_pose: dict[str, float],
    scene: pc.SceneLayout,
    rng: np.random.Generator | None,
) -> dict[str, float]:
    """根据杯子/机器人位置微调 approach 姿态，并加小扰动。"""
    return adapt_approach_tuning(base_pose, scene, rng)


def build_delta_approach_waypoints(
    start_arm: dict[str, float],
    forward_steps: int,
    down_steps: int,
    tuning: dict[str, float],
) -> list[dict[str, float]]:
    """离散 REACH 步进生成 approach 路点，避免插值轨迹碰杯。"""
    waypoints: list[dict[str, float]] = []
    pose = {joint: start_arm[joint] for joint in pc.KEY_CONTROL_JOINTS}
    for _ in range(forward_steps):
        for joint, delta in pc.REACH_FORWARD_DELTAS.items():
            pose[joint] = pose.get(joint, 0.0) + delta
        waypoints.append({**pose, **pc.GRIPPER_OPEN})
    for _ in range(down_steps):
        for joint, delta in pc.REACH_DOWN_DELTAS.items():
            pose[joint] = pose.get(joint, 0.0) + delta
        waypoints.append({**pose, **pc.GRIPPER_OPEN})
    final_pose = dict(pose)
    final_pose.update(
        {
            key: value
            for key, value in tuning.items()
            if key in pc.RIGHT_ARM_JOINTS
        }
    )
    elbow_delta = tuning.get("right_elbow_delta", 0.0)
    if abs(elbow_delta) > 1e-9:
        final_pose["right_elbow_joint"] = final_pose.get("right_elbow_joint", 0.0) + elbow_delta
    waypoints.append({**final_pose, **pc.GRIPPER_OPEN})
    return waypoints


def build_approach_waypoints(
    start_arm: dict[str, float],
    target_arm: dict[str, float],
    steps: int,
) -> list[dict[str, float]]:
    waypoints: list[dict[str, float]] = []
    joints = pc.KEY_CONTROL_JOINTS
    for step in range(1, steps + 1):
        alpha = step / steps
        waypoint = {
            joint: start_arm[joint] + alpha * (target_arm.get(joint, start_arm[joint]) - start_arm[joint])
            for joint in joints
        }
        waypoint.update(pc.GRIPPER_OPEN)
        waypoints.append(waypoint)
    return waypoints


def apply_alignment_correction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    teleop_geoms: frozenset[int],
    obstacle_geoms: frozenset[int],
) -> None:
    """闭合前根据夹爪-杯子几何关系做小步修正。"""
    mujoco.mj_collision(model, data)
    cup_pos = pc.get_cup_pose(model, data)[0]
    gripper_center = pc.get_gripper_center(model, data)
    dist = float(np.linalg.norm(cup_pos - gripper_center))
    (
        _,
        _,
        _,
        _,
        approach_dot,
        _dorsal_dot,
        _lateral_dot,
        finger_lat,
        _cup_above,
    ) = pc._grasp_alignment(model, data)

    if dist > 0.145:
        apply_arm_deltas(
            model,
            data,
            robot_ctrl,
            pc.REACH_FORWARD_DELTAS,
            teleop_geoms,
            obstacle_geoms,
        )
        return
    if dist < 0.118 and approach_dot > 0.45:
        apply_arm_deltas(
            model,
            data,
            robot_ctrl,
            pc.REACH_BACKWARD_DELTAS,
            teleop_geoms,
            obstacle_geoms,
        )
        return
    if approach_dot < pc.GRASP_MIN_APPROACH_DOT:
        apply_arm_deltas(
            model,
            data,
            robot_ctrl,
            pc.REACH_FORWARD_DELTAS,
            teleop_geoms,
            obstacle_geoms,
        )
        return
    if finger_lat > pc.GRASP_MAX_FINGER_LATERAL_OFFSET:
        lateral_error = float(cup_pos[0] - gripper_center[0])
        roll_delta = float(np.clip(-0.6 * lateral_error, -0.06, 0.06))
        pc.try_update_teleop_targets(
            model,
            data,
            robot_ctrl,
            {
                "right_shoulder_roll_joint": robot_ctrl["right_shoulder_roll_joint"]
                + roll_delta
            },
            teleop_geoms,
            obstacle_geoms,
        )
        return
    if dist > 0.125:
        apply_arm_deltas(
            model,
            data,
            robot_ctrl,
            pc.REACH_DOWN_DELTAS,
            teleop_geoms,
            obstacle_geoms,
        )


def apply_arm_deltas(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    deltas: dict[str, float],
    teleop_geoms: frozenset[int],
    obstacle_geoms: frozenset[int],
) -> bool:
    updates = {joint: robot_ctrl[joint] + delta for joint, delta in deltas.items()}
    return pc.try_update_teleop_targets(
        model,
        data,
        robot_ctrl,
        updates,
        teleop_geoms,
        obstacle_geoms,
    )


def sim_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    actuator_indices: dict[str, int],
    grasp: pc.GraspState,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> None:
    pc.apply_hold_ctrl(model, data, robot_ctrl, actuator_indices)
    pc.apply_kinematic_teleop(model, data, robot_ctrl)
    mujoco.mj_step(model, data)
    pc.apply_kinematic_teleop(model, data, robot_ctrl)
    pc.sanitize_cup_physics(model, data, grasp_active=grasp.active)
    pc.update_grasp(model, data, robot_ctrl, grasp, gripper_geoms, cup_geoms)
    mujoco.mj_forward(model, data)


def update_episode_grasp_metrics(
    metrics: EpisodeMetrics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    grasp: pc.GraspState,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
    *,
    sim_dt: float,
    initial_cup_z: float,
) -> None:
    """每步更新 episode 抓取指标（eval / RL 与 scripted 共用）。"""
    mujoco.mj_collision(model, data)
    cup_z = float(pc.get_cup_pose(model, data)[0][2])
    lift = cup_z - initial_cup_z
    metrics.max_cup_lift = max(metrics.max_cup_lift, lift)
    metrics.grasp_active = grasp.active

    if pc.is_gripper_closed(robot_ctrl) and pc.can_initiate_grasp(
        model,
        data,
        robot_ctrl,
        gripper_geoms,
        cup_geoms,
    ):
        metrics.valid_grasp_at_close = True
        metrics.had_contact_at_close = True

    if (
        grasp.active
        and pc.is_gripper_closed(robot_ctrl)
        and lift >= LIFT_HEIGHT_MIN
    ):
        if metrics.hold_min_lift <= 0.0:
            metrics.hold_min_lift = lift
        else:
            metrics.hold_min_lift = min(metrics.hold_min_lift, lift)
        if pc.can_maintain_grasp(
            model,
            data,
            robot_ctrl,
            gripper_geoms,
            cup_geoms,
        ):
            metrics.hold_valid_grasp_time += sim_dt


def check_episode_success(
    metrics: EpisodeMetrics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_ctrl: dict[str, float],
    grasp: pc.GraspState,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
) -> bool:
    """闭合时手心朝杯+有效接触，抬起>5cm，保持阶段全程合理抓取，终态仍有效。"""
    mujoco.mj_collision(model, data)
    metrics.grasp_active = grasp.active
    if not grasp.active:
        return False
    if not pc.is_gripper_closed(robot_ctrl):
        return False
    if not metrics.valid_grasp_at_close:
        return False
    if metrics.hold_min_lift < LIFT_HEIGHT_MIN:
        return False
    if metrics.hold_valid_grasp_time + 1e-6 < HOLD_TIME:
        return False
    return pc.can_initiate_grasp(
        model,
        data,
        robot_ctrl,
        gripper_geoms,
        cup_geoms,
    )


@dataclass
class SimContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    scene: pc.SceneLayout
    robot_ctrl: dict[str, float]
    actuator_indices: dict[str, int]
    teleop_geoms: frozenset[int]
    obstacle_geoms: frozenset[int]
    gripper_geoms: frozenset[int]
    cup_geoms: frozenset[int]
    grasp: pc.GraspState = field(default_factory=pc.GraspState)

    @classmethod
    def from_scene(cls, scene: pc.SceneLayout) -> SimContext:
        model, data = pc.make_model_data(scene)
        robot_ctrl = pc.INITIAL_QPOS.copy()
        actuator_indices = pc.get_actuator_indices(model)
        teleop_geoms, obstacle_geoms = pc.build_collision_geom_sets(model)
        gripper_geoms = pc.build_named_geom_set(model, pc.GRIPPER_GEOM_NAMES)
        cup_geoms = pc.build_named_geom_set(model, pc.CUP_GEOM_NAMES)
        grasp = pc.GraspState()
        pc.reset_scene(model, data, robot_ctrl, actuator_indices, scene=scene)
        return cls(
            model=model,
            data=data,
            scene=scene,
            robot_ctrl=robot_ctrl,
            actuator_indices=actuator_indices,
            teleop_geoms=teleop_geoms,
            obstacle_geoms=obstacle_geoms,
            gripper_geoms=gripper_geoms,
            cup_geoms=cup_geoms,
            grasp=grasp,
        )

    def reset(self) -> None:
        self.grasp.reset()
        self.robot_ctrl = pc.INITIAL_QPOS.copy()
        pc.reset_scene(
            self.model,
            self.data,
            self.robot_ctrl,
            self.actuator_indices,
            scene=self.scene,
        )

    def step(self) -> None:
        sim_step(
            self.model,
            self.data,
            self.robot_ctrl,
            self.actuator_indices,
            self.grasp,
            self.gripper_geoms,
            self.cup_geoms,
        )

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

    def apply_action_delta(self, delta: np.ndarray, *, scales: np.ndarray | None = None) -> bool:
        """对 8 维关节目标施加增量；若碰撞检测拒绝更新则返回 False。"""
        if scales is None:
            scaled = delta.astype(np.float64)
        else:
            scaled = delta.astype(np.float64) * scales
        updates = {
            name: self.robot_ctrl[name] + float(scaled[index])
            for index, name in enumerate(pc.ACTION_NAMES)
        }
        return pc.try_update_teleop_targets(
            self.model,
            self.data,
            self.robot_ctrl,
            updates,
            self.teleop_geoms,
            self.obstacle_geoms,
        )

    def cup_lift(self, initial_cup_z: float) -> float:
        cup_z = float(pc.get_cup_pose(self.model, self.data)[0][2])
        return cup_z - initial_cup_z

    def initial_cup_z(self) -> float:
        return float(pc.get_cup_pose(self.model, self.data)[0][2])


def run_scripted_episode(
    ctx: SimContext,
    controller: AutoPickController,
    renderer: mujoco.Renderer,
    recorder,
    *,
    record: bool,
    rng: np.random.Generator | None = None,
) -> bool:
    ctx.reset()
    controller.reset(
        {joint: ctx.robot_ctrl[joint] for joint in pc.KEY_CONTROL_JOINTS},
        ctx.scene,
        variant_index=controller.variant_index,
        rng=rng,
    )
    controller.bind_initial_cup_z(ctx.initial_cup_z())
    next_record_time = 0.0

    while not controller.finished:
        controller.advance(
            ctx.model.opt.timestep,
            ctx.model,
            ctx.data,
            ctx.robot_ctrl,
            ctx.grasp,
            ctx.teleop_geoms,
            ctx.obstacle_geoms,
            ctx.gripper_geoms,
            ctx.cup_geoms,
        )
        ctx.step()
        if record:
            next_record_time, _ = pc.maybe_record_frame(
                ctx.model,
                ctx.data,
                ctx.robot_ctrl,
                renderer,
                recorder,
                next_record_time,
            )

    return check_episode_success(
        controller.metrics,
        ctx.model,
        ctx.data,
        ctx.robot_ctrl,
        ctx.grasp,
        ctx.gripper_geoms,
        ctx.cup_geoms,
    )
