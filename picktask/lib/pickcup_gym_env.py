"""Gymnasium 抓杯环境（proprio 观测 + delta 动作）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import pickcup as pc
from lerobot_recorder import CAMERA_HEIGHT, CAMERA_WIDTH, RECORD_FPS
from pickcup_act_inference import ActInferenceBundle
from pickcup_reward import RewardConfig, compute_step_reward
from pickcup_sim_core import (
    HOLD_TIME,
    LIFT_HEIGHT_MIN,
    EpisodeMetrics,
    SimContext,
    check_episode_success,
    update_episode_grasp_metrics,
)

# 每步 delta 缩放：腰 + 右臂 5DOF + 双指夹爪
DEFAULT_DELTA_SCALES = np.array(
    [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.004, 0.004],
    dtype=np.float32,
)

# residual-v2：相对旧版再缩小 5×（总计 DEFAULT×0.1），夹爪通道锁死。
DEFAULT_RESIDUAL_SCALES = (DEFAULT_DELTA_SCALES * 0.1).copy()
DEFAULT_RESIDUAL_SCALES[-2:] = 0.0

# residual 观测 = proprio(8) + [gripper_closed, contact, lift, grasp_active]
RESIDUAL_OBS_EXTRA_DIM = 4


@dataclass
class PickCupEnvConfig:
    randomize_scene: bool = True
    control_hz: float = RECORD_FPS
    max_episode_steps: int = 900
    delta_scales: np.ndarray | None = None
    reward_config: RewardConfig | None = None
    drop_lift_threshold: float = 0.02
    cup_variant: str | None = None


class PickCupEnv(gym.Env):
    """MuJoCo 抓杯 RL 环境：obs=8 维 proprio，action=8 维归一化 delta。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: PickCupEnvConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or PickCupEnvConfig()
        self.cup_variant = pc.resolve_cup_variant(self.cfg.cup_variant)
        self.delta_scales = (
            self.cfg.delta_scales.copy()
            if self.cfg.delta_scales is not None
            else DEFAULT_DELTA_SCALES.copy()
        )
        self.reward_cfg = self.cfg.reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)

        self.ctx: SimContext | None = None
        self.renderer: mujoco.Renderer | None = None
        self.metrics = EpisodeMetrics()
        self.initial_cup_z = 0.0
        self.prev_lift = 0.0
        self.control_step = 0
        self.steps_per_control = 1
        self.sim_dt = 0.002

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(pc.PROPRIO_STATE_NAMES),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(pc.ACTION_NAMES),),
            dtype=np.float32,
        )

    def _ensure_sim(self, scene: pc.SceneLayout) -> None:
        needs_rebuild = self.ctx is None or self.ctx.scene.cup_variant != scene.cup_variant
        if needs_rebuild:
            self.ctx = SimContext.from_scene(scene)
            self.sim_dt = self.ctx.model.opt.timestep
            self.steps_per_control = max(1, int(round((1.0 / self.cfg.control_hz) / self.sim_dt)))
            self.renderer = None
        else:
            self.ctx.scene = scene
            self.ctx.reset()

    def _ensure_renderer(self) -> mujoco.Renderer:
        assert self.ctx is not None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
        return self.renderer

    def _get_obs(self) -> np.ndarray:
        assert self.ctx is not None
        return self.ctx.proprio().astype(np.float32)

    def _update_metrics(self) -> tuple[float, bool, bool, bool]:
        assert self.ctx is not None
        step_dt = self.ctx.model.opt.timestep * self.steps_per_control
        update_episode_grasp_metrics(
            self.metrics,
            self.ctx.model,
            self.ctx.data,
            self.ctx.robot_ctrl,
            self.ctx.grasp,
            self.ctx.gripper_geoms,
            self.ctx.cup_geoms,
            sim_dt=step_dt,
            initial_cup_z=self.initial_cup_z,
        )
        lift = self.ctx.cup_lift(self.initial_cup_z)
        mujoco.mj_collision(self.ctx.model, self.ctx.data)
        contact = pc.geoms_in_contact(
            self.ctx.data,
            self.ctx.gripper_geoms,
            self.ctx.cup_geoms,
        )
        dropped = lift < -self.cfg.drop_lift_threshold
        success = check_episode_success(
            self.metrics,
            self.ctx.model,
            self.ctx.data,
            self.ctx.robot_ctrl,
            self.ctx.grasp,
            self.ctx.gripper_geoms,
            self.ctx.cup_geoms,
        )
        return lift, contact, dropped, success

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        scene = (
            pc.sample_random_scene(self.rng, cup_variant=self.cup_variant)
            if self.cfg.randomize_scene
            else pc.SceneLayout.default(cup_variant=self.cup_variant)
        )
        self._ensure_sim(scene)
        assert self.ctx is not None

        self.metrics = EpisodeMetrics()
        self.initial_cup_z = self.ctx.initial_cup_z()
        self.metrics.initial_cup_z = self.initial_cup_z
        self.prev_lift = 0.0
        self.control_step = 0

        obs = self._get_obs()
        info = {
            "scene": scene,
            "initial_cup_z": self.initial_cup_z,
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.ctx is not None

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        rejected = not self.ctx.apply_action_delta(action, scales=self.delta_scales)

        for _ in range(self.steps_per_control):
            self.ctx.step()

        lift, contact, dropped, success = self._update_metrics()
        reward = compute_step_reward(
            prev_lift=self.prev_lift,
            curr_lift=lift,
            contact=contact,
            grasp_active=self.ctx.grasp.active,
            gripper_closed=pc.is_gripper_closed(self.ctx.robot_ctrl),
            had_contact_at_close=self.metrics.had_contact_at_close,
            action_delta=action * self.delta_scales,
            action_rejected=rejected,
            dropped=dropped,
            success=success,
            cfg=self.reward_cfg,
        )
        self.prev_lift = lift
        self.control_step += 1

        terminated = success or dropped
        truncated = self.control_step >= self.cfg.max_episode_steps
        if truncated and not terminated:
            success = check_episode_success(
                self.metrics,
                self.ctx.model,
                self.ctx.data,
                self.ctx.robot_ctrl,
                self.ctx.grasp,
                self.ctx.gripper_geoms,
                self.ctx.cup_geoms,
            )
            if success:
                reward += self.reward_cfg.success_bonus
                terminated = True

        obs = self._get_obs()
        info = {
            "lift": lift,
            "max_lift": self.metrics.max_cup_lift,
            "contact": contact,
            "grasp_active": self.ctx.grasp.active,
            "success": success,
            "action_rejected": rejected,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.ctx is None:
            return None
        return self.ctx.render_rgb(self._ensure_renderer())

    def close(self) -> None:
        self.renderer = None
        self.ctx = None


class ResidualActPickCupEnv(PickCupEnv):
    """在 ACT 关节目标上叠加 RL delta：a = a_ACT + delta（residual-v2）。"""

    def __init__(
        self,
        act_bundle: ActInferenceBundle,
        config: PickCupEnvConfig | None = None,
        *,
        seed: int | None = None,
        residual_scales: np.ndarray | None = None,
    ) -> None:
        cfg = config or PickCupEnvConfig()
        if cfg.reward_config is None:
            cfg = PickCupEnvConfig(
                randomize_scene=cfg.randomize_scene,
                control_hz=cfg.control_hz,
                max_episode_steps=cfg.max_episode_steps,
                delta_scales=cfg.delta_scales,
                reward_config=RewardConfig.for_residual(),
                drop_lift_threshold=cfg.drop_lift_threshold,
                cup_variant=cfg.cup_variant,
            )
        super().__init__(config=cfg, seed=seed)
        self.act_bundle = act_bundle
        self.residual_scales = (
            residual_scales.copy()
            if residual_scales is not None
            else DEFAULT_RESIDUAL_SCALES.copy()
        )
        self.prev_grasp_active = False
        self.prev_holding_5cm = False

        obs_dim = len(pc.PROPRIO_STATE_NAMES) + RESIDUAL_OBS_EXTRA_DIM
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _contact_now(self) -> bool:
        assert self.ctx is not None
        mujoco.mj_collision(self.ctx.model, self.ctx.data)
        return bool(
            pc.geoms_in_contact(
                self.ctx.data,
                self.ctx.gripper_geoms,
                self.ctx.cup_geoms,
            )
        )

    def _holding_5cm(self, lift: float) -> bool:
        assert self.ctx is not None
        return bool(
            self.ctx.grasp.active
            and pc.is_gripper_closed(self.ctx.robot_ctrl)
            and lift + 1e-9 >= LIFT_HEIGHT_MIN
        )

    def _get_obs(self) -> np.ndarray:
        assert self.ctx is not None
        proprio = self.ctx.proprio().astype(np.float32)
        lift = float(self.ctx.cup_lift(self.initial_cup_z))
        extras = np.array(
            [
                1.0 if pc.is_gripper_closed(self.ctx.robot_ctrl) else 0.0,
                1.0 if self._contact_now() else 0.0,
                lift,
                1.0 if self.ctx.grasp.active else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([proprio, extras], axis=0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.act_bundle.reset()
        self.prev_grasp_active = False
        self.prev_holding_5cm = False
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.ctx is not None

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        rgb = self.ctx.render_rgb(self._ensure_renderer())
        proprio = self.ctx.proprio()
        base_action = self.act_bundle.predict_action(rgb, proprio)
        combined = base_action + action * self.residual_scales
        rejected = not self.ctx.apply_action_vector(combined)

        for _ in range(self.steps_per_control):
            self.ctx.step()

        lift, contact, dropped, success = self._update_metrics()
        holding_5cm = self._holding_5cm(lift)
        grasp_lost = self.prev_grasp_active and not self.ctx.grasp.active
        hold_break = self.prev_holding_5cm and not holding_5cm

        reward = compute_step_reward(
            prev_lift=self.prev_lift,
            curr_lift=lift,
            contact=contact,
            grasp_active=self.ctx.grasp.active,
            gripper_closed=pc.is_gripper_closed(self.ctx.robot_ctrl),
            had_contact_at_close=self.metrics.had_contact_at_close,
            action_delta=action * self.residual_scales,
            action_rejected=rejected,
            dropped=dropped,
            success=success,
            cfg=self.reward_cfg,
            grasp_lost=grasp_lost,
            holding_5cm=holding_5cm,
            hold_break=hold_break,
        )
        self.prev_lift = lift
        self.prev_grasp_active = bool(self.ctx.grasp.active)
        self.prev_holding_5cm = holding_5cm
        self.control_step += 1

        terminated = success or dropped
        truncated = self.control_step >= self.cfg.max_episode_steps
        if truncated and not terminated:
            success = check_episode_success(
                self.metrics,
                self.ctx.model,
                self.ctx.data,
                self.ctx.robot_ctrl,
                self.ctx.grasp,
                self.ctx.gripper_geoms,
                self.ctx.cup_geoms,
            )
            if success:
                reward += self.reward_cfg.success_bonus
                terminated = True

        obs = self._get_obs()
        info = {
            "lift": lift,
            "max_lift": self.metrics.max_cup_lift,
            "contact": contact,
            "grasp_active": self.ctx.grasp.active,
            "holding_5cm": holding_5cm,
            "success": success,
            "action_rejected": rejected,
            "base_action": base_action,
            "grasp_lost": grasp_lost,
            "hold_break": hold_break,
        }
        return obs, reward, terminated, truncated, info
