#!/usr/bin/env python3
"""Residual RL 细粒度分层评估（对齐 ACT granular，并增加更详细指标）。

对比：
  - act_only: 纯 ACT（残差=0）
  - residual_best / residual_final / 可选 mid ckpt

分层指标与 ACT 一致：接触 / 抓握 / 抬≥1cm / 保持 500ms / 1.5s；
额外：抬≥5cm、标准成功、残差动作统计、首次接触步数等。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# macOS: 必须先 import torch，再 import numpy。
import torch
import numpy as np

from _bootstrap import bootstrap

PICKTASK_DIR = bootstrap(include_lerobot=False)

LIFT_1CM = 0.01
LIFT_5CM = 0.05
HOLD_500MS = 0.5
HOLD_1500MS = 1.5

# residual-v2 scales（与 pickcup_gym_env.DEFAULT_RESIDUAL_SCALES 保持一致；
# 勿在模块顶层 import pickcup_gym_env，以免 macOS 上先于 ACT 拉入 mujoco）。
RESIDUAL_SCALES = np.array(
    [0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0, 0.0],
    dtype=np.float32,
)

MILESTONE_NAMES = (
    "touch_cup",
    "grasp",
    "lift_1cm",
    "lift_5cm",
    "hold_500ms",
    "hold_1500ms",
    "standard_success",
)

MILESTONE_LABELS = {
    "touch_cup": "接触杯子",
    "grasp": "有效抓握",
    "lift_1cm": "抬起≥1cm",
    "lift_5cm": "抬起≥5cm",
    "hold_500ms": "保持≥500ms(≥1cm)",
    "hold_1500ms": "保持≥1.5s(≥1cm)",
    "standard_success": "标准成功(≥5cm+1.5s)",
}


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class DetailedEpisodeMetrics:
    touched_cup: bool = False
    valid_grasp: bool = False
    grasp_active: bool = False
    max_cup_lift: float = 0.0
    hold_time_at_1cm: float = 0.0
    hold_time_at_5cm: float = 0.0
    first_touch_control_step: int | None = None
    first_grasp_control_step: int | None = None
    standard_success: bool = False
    mean_abs_residual: float = 0.0
    max_abs_residual: float = 0.0
    residual_l2_mean: float = 0.0
    n_control_steps: int = 0

    @property
    def touch_cup(self) -> bool:
        return self.touched_cup

    @property
    def grasp(self) -> bool:
        return self.valid_grasp or self.grasp_active

    @property
    def lift_1cm(self) -> bool:
        return self.max_cup_lift + 1e-9 >= LIFT_1CM

    @property
    def lift_5cm(self) -> bool:
        return self.max_cup_lift + 1e-9 >= LIFT_5CM

    @property
    def hold_500ms(self) -> bool:
        return self.hold_time_at_1cm + 1e-6 >= HOLD_500MS

    @property
    def hold_1500ms(self) -> bool:
        return self.hold_time_at_1cm + 1e-6 >= HOLD_1500MS

    def milestones(self) -> dict[str, bool]:
        return {
            "touch_cup": self.touch_cup,
            "grasp": self.grasp,
            "lift_1cm": self.lift_1cm,
            "lift_5cm": self.lift_5cm,
            "hold_500ms": self.hold_500ms,
            "hold_1500ms": self.hold_1500ms,
            "standard_success": self.standard_success,
        }


@dataclass
class PolicyEvalSummary:
    name: str
    policy_path: str | None
    episodes: int
    counts: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    per_episode: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    mean_max_lift_m: float = 0.0
    mean_hold_1cm_s: float = 0.0
    mean_abs_residual: float = 0.0

    def compute_aggregates(self) -> None:
        denom = max(self.episodes, 1)
        self.rates = {name: self.counts.get(name, 0) / denom for name in MILESTONE_NAMES}
        if self.per_episode:
            self.mean_max_lift_m = float(np.mean([e["max_cup_lift_m"] for e in self.per_episode]))
            self.mean_hold_1cm_s = float(np.mean([e["hold_time_at_1cm_s"] for e in self.per_episode]))
            self.mean_abs_residual = float(np.mean([e["mean_abs_residual"] for e in self.per_episode]))


def update_detailed_metrics(
    metrics: DetailedEpisodeMetrics,
    *,
    model,
    data,
    robot_ctrl: dict[str, float],
    grasp,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
    sim_dt: float,
    initial_cup_z: float,
    control_step: int,
) -> None:
    import mujoco

    import pickcup as pc

    mujoco.mj_collision(model, data)

    if pc.geoms_in_contact(data, gripper_geoms, cup_geoms):
        if not metrics.touched_cup:
            metrics.first_touch_control_step = control_step
        metrics.touched_cup = True

    cup_z = float(pc.get_cup_pose(model, data)[0][2])
    lift = cup_z - initial_cup_z
    metrics.max_cup_lift = max(metrics.max_cup_lift, lift)

    if grasp.active:
        metrics.grasp_active = True

    if pc.is_gripper_closed(robot_ctrl) and pc.can_initiate_grasp(
        model,
        data,
        robot_ctrl,
        gripper_geoms,
        cup_geoms,
    ):
        if not metrics.valid_grasp:
            metrics.first_grasp_control_step = control_step
        metrics.valid_grasp = True

    holding_1cm = (
        grasp.active
        and pc.is_gripper_closed(robot_ctrl)
        and lift + 1e-9 >= LIFT_1CM
        and pc.can_maintain_grasp(model, data, robot_ctrl, gripper_geoms, cup_geoms)
    )
    if holding_1cm:
        metrics.hold_time_at_1cm += sim_dt

    holding_5cm = (
        grasp.active
        and pc.is_gripper_closed(robot_ctrl)
        and lift + 1e-9 >= LIFT_5CM
        and pc.can_maintain_grasp(model, data, robot_ctrl, gripper_geoms, cup_geoms)
    )
    if holding_5cm:
        metrics.hold_time_at_5cm += sim_dt


def build_residual_obs(ctx, initial_cup_z: float) -> np.ndarray:
    """与 ResidualACTPickCupEnv._get_obs 对齐：proprio(8)+extras(4)=12。"""
    import mujoco
    import pickcup as pc

    proprio = ctx.proprio().astype(np.float32)
    lift = float(ctx.cup_lift(initial_cup_z))
    mujoco.mj_collision(ctx.model, ctx.data)
    contact = pc.geoms_in_contact(
        ctx.data,
        ctx.gripper_geoms,
        ctx.cup_geoms,
    )
    extras = np.array(
        [
            1.0 if pc.is_gripper_closed(ctx.robot_ctrl) else 0.0,
            1.0 if contact else 0.0,
            lift,
            1.0 if ctx.grasp.active else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([proprio, extras], axis=0)


def residual_policy_obs(ctx, initial_cup_z: float, residual_agent) -> np.ndarray:
    """按 checkpoint 的 obs_dim 组装 residual 输入（v1=8 / v2=12）。"""
    proprio = ctx.proprio().astype(np.float32)
    if residual_agent is None:
        return proprio
    obs_dim = int(residual_agent.cfg.obs_dim)
    if obs_dim == len(proprio):
        return proprio
    if obs_dim == len(proprio) + 4:
        return build_residual_obs(ctx, initial_cup_z)
    raise ValueError(
        f"residual obs_dim={obs_dim} 与 proprio={len(proprio)} 不兼容（期望 8 或 12）"
    )


def run_detailed_episode(
    ctx,
    renderer,
    act_bundle,
    residual_agent,
    *,
    max_steps: int,
    control_hz: float,
    residual_scales: np.ndarray,
) -> DetailedEpisodeMetrics:
    import pickcup as pc
    from pickcup_sim_core import EpisodeMetrics, check_episode_success, update_episode_grasp_metrics

    ctx.reset()
    act_bundle.reset()
    metrics = DetailedEpisodeMetrics()
    std_metrics = EpisodeMetrics()
    std_metrics.initial_cup_z = ctx.initial_cup_z()
    initial_cup_z = std_metrics.initial_cup_z

    control_dt = 1.0 / control_hz
    sim_dt = ctx.model.opt.timestep
    steps_per_control = max(1, int(round(control_dt / sim_dt)))

    abs_res_sum = 0.0
    abs_res_max = 0.0
    l2_sum = 0.0
    n_ctrl = 0

    for step in range(max_steps):
        if step % steps_per_control == 0:
            rgb = ctx.render_rgb(renderer)
            proprio = ctx.proprio()
            base = act_bundle.predict_action(rgb, proprio)
            if residual_agent is None:
                delta = np.zeros_like(base)
            else:
                obs = residual_policy_obs(ctx, initial_cup_z, residual_agent)
                delta = residual_agent.predict(obs, deterministic=True).astype(np.float32)
            delta = np.clip(delta, -1.0, 1.0)
            applied_delta = delta * residual_scales
            action = base + applied_delta
            ctx.apply_action_vector(action)

            abs_mean = float(np.mean(np.abs(delta)))
            abs_res_sum += abs_mean
            abs_res_max = max(abs_res_max, float(np.max(np.abs(delta))))
            l2_sum += float(np.linalg.norm(delta))
            n_ctrl += 1
            control_step = n_ctrl
        else:
            control_step = n_ctrl

        ctx.step()
        update_detailed_metrics(
            metrics,
            model=ctx.model,
            data=ctx.data,
            robot_ctrl=ctx.robot_ctrl,
            grasp=ctx.grasp,
            gripper_geoms=ctx.gripper_geoms,
            cup_geoms=ctx.cup_geoms,
            sim_dt=sim_dt,
            initial_cup_z=initial_cup_z,
            control_step=control_step,
        )
        update_episode_grasp_metrics(
            std_metrics,
            ctx.model,
            ctx.data,
            ctx.robot_ctrl,
            ctx.grasp,
            ctx.gripper_geoms,
            ctx.cup_geoms,
            sim_dt=sim_dt,
            initial_cup_z=initial_cup_z,
        )
        if check_episode_success(
            std_metrics,
            ctx.model,
            ctx.data,
            ctx.robot_ctrl,
            ctx.grasp,
            ctx.gripper_geoms,
            ctx.cup_geoms,
        ):
            metrics.standard_success = True

    metrics.n_control_steps = n_ctrl
    if n_ctrl > 0:
        metrics.mean_abs_residual = abs_res_sum / n_ctrl
        metrics.max_abs_residual = abs_res_max
        metrics.residual_l2_mean = l2_sum / n_ctrl
    return metrics


def evaluate_policy(
    name: str,
    eval_scenes: list[tuple[object, str]],
    *,
    act_bundle,
    residual_agent,
    policy_path: Path | None,
    max_steps: int,
    verbose: bool,
) -> PolicyEvalSummary:
    import mujoco

    from lerobot_recorder import CAMERA_HEIGHT, CAMERA_WIDTH, RECORD_FPS
    from pickcup_sim_core import SimContext

    summary = PolicyEvalSummary(
        name=name,
        policy_path=str(policy_path) if policy_path else None,
        episodes=len(eval_scenes),
    )
    summary.counts = {n: 0 for n in MILESTONE_NAMES}
    t0 = time.time()

    for index, (scene, scene_label) in enumerate(eval_scenes):
        ctx = SimContext.from_scene(scene)
        renderer = mujoco.Renderer(ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
        ep_t0 = time.time()
        metrics = run_detailed_episode(
            ctx,
            renderer,
            act_bundle,
            residual_agent,
            max_steps=max_steps,
            control_hz=RECORD_FPS,
            residual_scales=RESIDUAL_SCALES,
        )
        ep_elapsed = time.time() - ep_t0
        milestones = metrics.milestones()
        for key, ok in milestones.items():
            if ok:
                summary.counts[key] += 1

        record = {
            "episode": index + 1,
            "scene": scene_label,
            "elapsed_s": round(ep_elapsed, 2),
            "max_cup_lift_m": round(metrics.max_cup_lift, 4),
            "hold_time_at_1cm_s": round(metrics.hold_time_at_1cm, 3),
            "hold_time_at_5cm_s": round(metrics.hold_time_at_5cm, 3),
            "first_touch_control_step": metrics.first_touch_control_step,
            "first_grasp_control_step": metrics.first_grasp_control_step,
            "mean_abs_residual": round(metrics.mean_abs_residual, 4),
            "max_abs_residual": round(metrics.max_abs_residual, 4),
            "residual_l2_mean": round(metrics.residual_l2_mean, 4),
            "n_control_steps": metrics.n_control_steps,
            **milestones,
        }
        summary.per_episode.append(record)

        if verbose:
            flags = " ".join(f"{k[:6]}={'Y' if v else 'N'}" for k, v in milestones.items())
            print(
                f"  [{index + 1:2d}/{len(eval_scenes)}] lift={metrics.max_cup_lift:.3f}m "
                f"hold1={metrics.hold_time_at_1cm:.2f}s |Δ|={metrics.mean_abs_residual:.3f} "
                f"{flags}"
            )

    summary.elapsed_s = time.time() - t0
    summary.compute_aggregates()
    return summary


def format_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({count / max(total, 1) * 100:.1f}%)"


def print_summary_table(summaries: list[PolicyEvalSummary]) -> None:
    episodes = summaries[0].episodes if summaries else 0
    print("\n" + "=" * 96)
    print(f"Residual RL 细粒度评估汇总（{episodes} episodes / policy）")
    print("=" * 96)
    header = f"{'Policy':>18}"
    for name in MILESTONE_NAMES:
        header += f" | {MILESTONE_LABELS[name]}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = f"{s.name:>18}"
        for name in MILESTONE_NAMES:
            row += f" | {format_rate(s.counts.get(name, 0), s.episodes)}"
        print(row)
    print("-" * len(header))
    print("附加均值：")
    for s in summaries:
        print(
            f"  {s.name:>18}: mean_lift={s.mean_max_lift_m:.4f}m  "
            f"mean_hold@1cm={s.mean_hold_1cm_s:.3f}s  "
            f"mean_|Δ|={s.mean_abs_residual:.4f}  "
            f"time={s.elapsed_s:.1f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual RL 细粒度分层评估")
    parser.add_argument(
        "--act-policy",
        type=Path,
        default=Path("outputs/train/act_pickcup_train_200/checkpoints/step_080000"),
    )
    parser.add_argument(
        "--act-dataset",
        type=Path,
        default=PICKTASK_DIR / "pickcupdata" / "train" / "pickcup_train_200",
    )
    parser.add_argument(
        "--rl-dir",
        type=Path,
        default=Path("outputs/rl/residual_act_80k"),
        help="含 best/final/checkpoints 的 RL 输出目录",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="act_only,residual_best,residual_final",
        help="逗号分隔：act_only,residual_best,residual_final,residual_XXXXXX",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3000, help="仿真步数（与 ACT granular 一致）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--scene-source", choices=["success-log", "random", "default"], default="success-log")
    parser.add_argument("--scenes-log", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PICKTASK_DIR / "outputs" / "rl_granular_eval_30ep.json",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_policy_specs(names: list[str], rl_dir: Path) -> list[tuple[str, Path | None]]:
    specs: list[tuple[str, Path | None]] = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        if name == "act_only":
            specs.append((name, None))
        elif name == "residual_best":
            specs.append((name, rl_dir / "ppo_pickcup_best.pt"))
        elif name == "residual_final":
            specs.append((name, rl_dir / "ppo_pickcup_final.pt"))
        elif name.startswith("residual_"):
            tag = name[len("residual_") :]
            path = rl_dir / "checkpoints" / f"ppo_pickcup_{tag}.pt"
            if not path.exists():
                # allow residual_000095 style already with zeros
                path = rl_dir / "checkpoints" / f"ppo_pickcup_{int(tag):06d}.pt"
            specs.append((name, path))
        else:
            raise ValueError(f"未知 policy 名: {name}")
    return specs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    device = pick_device(args.device)
    act_policy = args.act_policy.resolve()
    act_dataset = args.act_dataset.resolve()
    rl_dir = args.rl_dir.resolve()
    policy_specs = resolve_policy_specs(args.policies.split(","), rl_dir)

    for name, path in policy_specs:
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{name} checkpoint 不存在: {path}")

    # macOS: ACT from_pretrained 必须在 import mujoco / gymnasium 之前。
    bootstrap(include_lerobot=True)
    from pickcup_act_inference import load_act_inference_bundle

    print(f"加载 ACT: {act_policy}", flush=True)
    act_bundle = load_act_inference_bundle(act_policy, act_dataset, device)
    print("ACT 加载完成", flush=True)

    from pickcup_ppo import PPOAgent
    from pickcup_sim_eval import build_eval_scenes
    import pickcup as pc

    eval_scenes, scene_source_label = build_eval_scenes(
        pc_module=pc,
        dataset_root=act_dataset,
        scenes_log=args.scenes_log.resolve() if args.scenes_log else None,
        scene_source=args.scene_source,
        episodes=args.episodes,
        seed=args.seed,
    )

    print("=" * 78)
    print("Residual RL 细粒度分层评估（详细版）")
    print("=" * 78)
    print(f"  act_policy:  {act_policy}")
    print(f"  act_dataset: {act_dataset}")
    print(f"  rl_dir:      {rl_dir}")
    print(f"  policies:    {', '.join(n for n, _ in policy_specs)}")
    print(f"  episodes:    {args.episodes} (seed={args.seed})")
    print(f"  scene:       {scene_source_label}")
    print(f"  max_steps:   {args.max_steps}")
    print(f"  device:      {device}")
    print("=" * 78)

    summaries: list[PolicyEvalSummary] = []
    for name, path in policy_specs:
        print(f"\n>>> 评估 {name}" + (f" ({path.name})" if path else " (Δ=0)") + " ...", flush=True)
        agent = None if path is None else PPOAgent.load(path, device=torch.device("cpu"))
        if agent is not None:
            print(f"    residual obs_dim={agent.cfg.obs_dim} action_dim={agent.cfg.action_dim}", flush=True)
        summary = evaluate_policy(
            name,
            eval_scenes,
            act_bundle=act_bundle,
            residual_agent=agent,
            policy_path=path,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        summaries.append(summary)
        print(
            f"    完成 {summary.elapsed_s:.1f}s | "
            f"touch={summary.rates['touch_cup']*100:.0f}% "
            f"grasp={summary.rates['grasp']*100:.0f}% "
            f"std_ok={summary.rates['standard_success']*100:.0f}%",
            flush=True,
        )

    print_summary_table(summaries)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": args.episodes,
            "seed": args.seed,
            "scene_source": scene_source_label,
            "max_steps": args.max_steps,
            "act_policy": str(act_policy),
            "act_dataset": str(act_dataset),
            "rl_dir": str(rl_dir),
            "milestones": MILESTONE_LABELS,
            "residual_scales": RESIDUAL_SCALES.tolist(),
            "policies": [
                {
                    "name": s.name,
                    "policy_path": s.policy_path,
                    "counts": s.counts,
                    "rates": s.rates,
                    "mean_max_lift_m": s.mean_max_lift_m,
                    "mean_hold_1cm_s": s.mean_hold_1cm_s,
                    "mean_abs_residual": s.mean_abs_residual,
                    "elapsed_s": s.elapsed_s,
                    "per_episode": s.per_episode,
                }
                for s in summaries
            ],
        }
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\n已写入 {args.output_json}")


if __name__ == "__main__":
    main()
