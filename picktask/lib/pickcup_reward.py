"""抓杯 RL 稠密 / 稀疏 reward 计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    lift_scale: float = 20.0
    lift_delta_scale: float = 50.0
    contact_bonus: float = 0.02
    grasp_bonus: float = 0.05
    close_with_contact_bonus: float = 0.03
    success_bonus: float = 10.0
    step_penalty: float = -0.001
    drop_penalty: float = -2.0
    collision_penalty: float = -0.05
    action_penalty_scale: float = 0.01
    # residual-v2：保护抓取 / 保持（纯 PPO 默认关闭）
    grasp_lost_penalty: float = 0.0
    hold_shaping_scale: float = 0.0
    hold_break_penalty: float = 0.0

    @classmethod
    def for_residual(cls) -> RewardConfig:
        """residual PPO：少刷假抬升、惩罚大残差、保护 hold。"""
        return cls(
            lift_scale=5.0,
            lift_delta_scale=10.0,
            contact_bonus=0.02,
            grasp_bonus=0.08,
            close_with_contact_bonus=0.05,
            success_bonus=10.0,
            step_penalty=-0.001,
            drop_penalty=-3.0,
            collision_penalty=-0.05,
            action_penalty_scale=0.2,
            grasp_lost_penalty=1.5,
            hold_shaping_scale=0.05,
            hold_break_penalty=2.0,
        )


def compute_step_reward(
    *,
    prev_lift: float,
    curr_lift: float,
    contact: bool,
    grasp_active: bool,
    gripper_closed: bool,
    had_contact_at_close: bool,
    action_delta: np.ndarray,
    action_rejected: bool,
    dropped: bool,
    success: bool,
    cfg: RewardConfig,
    grasp_lost: bool = False,
    holding_5cm: bool = False,
    hold_break: bool = False,
) -> float:
    reward = cfg.step_penalty

    lift_delta = curr_lift - prev_lift
    reward += cfg.lift_delta_scale * lift_delta
    reward += cfg.lift_scale * max(curr_lift, 0.0) * 0.01

    if contact:
        reward += cfg.contact_bonus
    if grasp_active:
        reward += cfg.grasp_bonus
    if gripper_closed and (contact or had_contact_at_close):
        reward += cfg.close_with_contact_bonus

    if holding_5cm:
        reward += cfg.hold_shaping_scale
    if grasp_lost:
        reward -= cfg.grasp_lost_penalty
    if hold_break:
        reward -= cfg.hold_break_penalty

    if action_rejected:
        reward += cfg.collision_penalty

    if dropped:
        reward += cfg.drop_penalty

    if success:
        reward += cfg.success_bonus

    norm = float(np.linalg.norm(action_delta))
    reward -= cfg.action_penalty_scale * norm
    return reward
