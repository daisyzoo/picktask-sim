#!/usr/bin/env python3
"""多 checkpoint 细粒度 ACT 评估：接触 / 抓握 / 抬升 / 保持 等分层成功率。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from _bootstrap import bootstrap

bootstrap(include_lerobot=True)

if TYPE_CHECKING:
    import mujoco

    from pickcup_sim_core import SimContext

# 与 pickcup_sim_eval 共用场景构造与策略加载。
from pickcup_sim_eval import build_eval_scenes, load_policy_and_processors

LIFT_1CM = 0.01
HOLD_500MS = 0.5
HOLD_1500MS = 1.5

MILESTONE_NAMES = (
    "touch_cup",
    "grasp",
    "lift_1cm",
    "hold_500ms",
    "hold_1500ms",
)

MILESTONE_LABELS = {
    "touch_cup": "接触杯子",
    "grasp": "有效抓握",
    "lift_1cm": "抬起≥1cm",
    "hold_500ms": "保持≥500ms(≥1cm)",
    "hold_1500ms": "保持≥1.5s(≥1cm)",
}


@dataclass
class GranularEpisodeMetrics:
    """单 episode 分层指标（全程累计，不因达标提前结束）。"""

    touched_cup: bool = False
    valid_grasp: bool = False
    grasp_active: bool = False
    max_cup_lift: float = 0.0
    hold_time_at_1cm: float = 0.0

    @property
    def touch_cup(self) -> bool:
        return self.touched_cup

    @property
    def grasp(self) -> bool:
        # 闭合时满足 can_initiate_grasp，或物理吸附 grasp.active 任一成立。
        return self.valid_grasp or self.grasp_active

    @property
    def lift_1cm(self) -> bool:
        return self.max_cup_lift + 1e-9 >= LIFT_1CM

    @property
    def hold_500ms(self) -> bool:
        return self.hold_time_at_1cm + 1e-6 >= HOLD_500MS

    @property
    def hold_1500ms(self) -> bool:
        return self.hold_time_at_1cm + 1e-6 >= HOLD_1500MS

    def milestones(self) -> dict[str, bool]:
        return {name: getattr(self, name) for name in MILESTONE_NAMES}


@dataclass
class GranularEvalSummary:
    step: int
    episodes: int
    counts: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    per_episode: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0

    def compute_rates(self) -> None:
        denom = max(self.episodes, 1)
        self.rates = {
            name: self.counts.get(name, 0) / denom for name in MILESTONE_NAMES
        }


def update_granular_metrics(
    metrics: GranularEpisodeMetrics,
    *,
    model: object,
    data: object,
    robot_ctrl: dict[str, float],
    grasp: object,
    gripper_geoms: frozenset[int],
    cup_geoms: frozenset[int],
    sim_dt: float,
    initial_cup_z: float,
) -> None:
    import mujoco

    import pickcup as pc

    mujoco.mj_collision(model, data)

    if pc.geoms_in_contact(data, gripper_geoms, cup_geoms):
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
        metrics.valid_grasp = True

    holding = (
        grasp.active
        and pc.is_gripper_closed(robot_ctrl)
        and lift + 1e-9 >= LIFT_1CM
        and pc.can_maintain_grasp(
            model,
            data,
            robot_ctrl,
            gripper_geoms,
            cup_geoms,
        )
    )
    if holding:
        metrics.hold_time_at_1cm += sim_dt


def run_granular_episode(
    ctx: SimContext,
    renderer: mujoco.Renderer,
    policy,
    metadata,
    preprocess,
    postprocess,
    build_inference_frame,
    make_robot_action,
    device: object,
    *,
    max_steps: int,
    control_hz: float,
    action_names: tuple[str, ...],
    proprio_state_names: tuple[str, ...],
    default_task: str,
) -> GranularEpisodeMetrics:
    import numpy as np

    ctx.reset()
    policy.reset()
    metrics = GranularEpisodeMetrics()
    initial_cup_z = ctx.initial_cup_z()
    control_dt = 1.0 / control_hz
    sim_dt = ctx.model.opt.timestep
    steps_per_control = max(1, int(round(control_dt / sim_dt)))

    for step in range(max_steps):
        if step % steps_per_control == 0:
            rgb = ctx.render_rgb(renderer)
            proprio = ctx.proprio()
            obs_dict = {
                "head_camera": rgb,
                **{
                    name: float(proprio[index])
                    for index, name in enumerate(proprio_state_names)
                },
            }
            obs_frame = build_inference_frame(
                observation=obs_dict,
                device=device,
                ds_features=metadata.features,
                task=default_task,
            )
            batch = preprocess(obs_frame)
            action_tensor = policy.select_action(batch)
            action_tensor = postprocess(action_tensor)
            action_dict = make_robot_action(action_tensor, metadata.features)
            action_vec = np.array(
                [action_dict[name] for name in action_names],
                dtype=np.float32,
            )
            ctx.apply_action_vector(action_vec)

        ctx.step()
        update_granular_metrics(
            metrics,
            model=ctx.model,
            data=ctx.data,
            robot_ctrl=ctx.robot_ctrl,
            grasp=ctx.grasp,
            gripper_geoms=ctx.gripper_geoms,
            cup_geoms=ctx.cup_geoms,
            sim_dt=sim_dt,
            initial_cup_z=initial_cup_z,
        )

    return metrics


def evaluate_checkpoint(
    policy_path: Path,
    eval_scenes: list[tuple[object, str]],
    *,
    step: int,
    dataset_root: Path,
    device_name: str,
    max_steps: int,
    verbose: bool,
) -> GranularEvalSummary:
    (
        policy,
        metadata,
        preprocess,
        postprocess,
        build_inference_frame,
        make_robot_action,
        device,
    ) = load_policy_and_processors(policy_path, dataset_root, device_name)

    import mujoco

    import pickcup as pc
    from lerobot_recorder import CAMERA_HEIGHT, CAMERA_WIDTH, DEFAULT_TASK, RECORD_FPS
    from pickcup_sim_core import SimContext

    summary = GranularEvalSummary(step=step, episodes=len(eval_scenes))
    summary.counts = {name: 0 for name in MILESTONE_NAMES}
    t0 = time.time()

    for index, (scene, scene_label) in enumerate(eval_scenes):
        ctx = SimContext.from_scene(scene)
        renderer = mujoco.Renderer(ctx.model, CAMERA_HEIGHT, CAMERA_WIDTH)
        ep_t0 = time.time()
        metrics = run_granular_episode(
            ctx,
            renderer,
            policy,
            metadata,
            preprocess,
            postprocess,
            build_inference_frame,
            make_robot_action,
            device,
            max_steps=max_steps,
            control_hz=RECORD_FPS,
            action_names=pc.ACTION_NAMES,
            proprio_state_names=pc.PROPRIO_STATE_NAMES,
            default_task=DEFAULT_TASK,
        )
        ep_elapsed = time.time() - ep_t0
        milestones = metrics.milestones()
        for name, ok in milestones.items():
            if ok:
                summary.counts[name] += 1

        episode_record = {
            "episode": index + 1,
            "scene": scene_label,
            "elapsed_s": round(ep_elapsed, 2),
            "max_cup_lift_m": round(metrics.max_cup_lift, 4),
            "hold_time_at_1cm_s": round(metrics.hold_time_at_1cm, 3),
            **milestones,
        }
        summary.per_episode.append(episode_record)

        if verbose:
            short = {
                "touch_cup": "touch",
                "grasp": "grasp",
                "lift_1cm": "lift1",
                "hold_500ms": "h500",
                "hold_1500ms": "h1500",
            }
            flags = " ".join(
                f"{short[name]}={'Y' if milestones[name] else 'N'}"
                for name in MILESTONE_NAMES
            )
            print(
                f"  [{index + 1:2d}/{len(eval_scenes)}] lift={metrics.max_cup_lift:.3f}m "
                f"hold@1cm={metrics.hold_time_at_1cm:.2f}s {flags} {scene_label}"
            )

    summary.elapsed_s = time.time() - t0
    summary.compute_rates()
    return summary


def format_rate(count: int, total: int) -> str:
    rate = count / max(total, 1) * 100.0
    return f"{count}/{total} ({rate:.1f}%)"


def print_summary_table(summaries: list[GranularEvalSummary]) -> None:
    episodes = summaries[0].episodes if summaries else 0
    print("\n" + "=" * 78)
    print(f"细粒度评估汇总（{episodes} episodes / checkpoint）")
    print("=" * 78)
    header = f"{'Checkpoint':>12}"
    for name in MILESTONE_NAMES:
        header += f" | {MILESTONE_LABELS[name]:^16}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        row = f"step_{summary.step:06d}"
        for name in MILESTONE_NAMES:
            count = summary.counts[name]
            rate = summary.rates[name] * 100.0
            row += f" | {count:2d}/{summary.episodes} ({rate:4.1f}%)"
        print(row)
    print("=" * 78)
    print("指标定义:")
    print("  接触杯子     : 夹爪与杯体 geom 发生接触 (dist≤2mm)")
    print("  有效抓握     : 闭合时 can_initiate_grasp，或 grasp.active 吸附")
    print("  抬起≥1cm     : episode 内 max_cup_lift ≥ 1cm")
    print("  保持≥500ms   : 在 lift≥1cm 且 grasp 有效时累计 hold ≥ 0.5s")
    print("  保持≥1.5s    : 同上，累计 hold ≥ 1.5s")


def parse_steps(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("至少指定一个训练步数")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多 checkpoint 细粒度 ACT 抓杯评估（接触/抓握/抬升/保持）",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train/act_pickcup_train_200/checkpoints"),
        help="checkpoint 根目录，内含 step_XXXXXX 子目录",
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default="30000,50000,70000,80000",
        help="逗号分隔的训练步数，如 30000,50000,70000,80000",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("picktask/pickcupdata/train/pickcup_train_200"),
        help="训练 dataset 根目录（stats/features）",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--scene-source",
        choices=["auto", "success-log", "random"],
        default="auto",
    )
    parser.add_argument("--scenes-log", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选：将完整结果写入 JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个 episode 明细",
    )
    return parser.parse_args()


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    dataset_root = args.dataset.resolve()
    train_dir = args.train_dir.resolve()

    first_policy = train_dir / f"step_{args.steps[0]:06d}"
    if not first_policy.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {first_policy}")

    # macOS: 须先加载 ACT，再 import mujoco / pickcup，否则可能 segfault。
    _warmup_policy, *_ = load_policy_and_processors(
        first_policy,
        dataset_root,
        args.device,
    )
    del _warmup_policy

    import pickcup as pc

    eval_scenes, scene_source_label = build_eval_scenes(
        pc_module=pc,
        dataset_root=dataset_root,
        scenes_log=args.scenes_log.resolve() if args.scenes_log else None,
        scene_source=args.scene_source,
        episodes=args.episodes,
        seed=args.seed,
    )

    print("=" * 78)
    print("ACT 细粒度分层评估")
    print("=" * 78)
    print(f"  checkpoints: {', '.join(f'step_{s:06d}' for s in args.steps)}")
    print(f"  train dir:   {train_dir}")
    print(f"  dataset:     {dataset_root}")
    print(f"  episodes:    {args.episodes} (seed={args.seed})")
    print(f"  scene:       {scene_source_label}")
    print(f"  max steps:   {args.max_steps}")
    print(f"  device:      {args.device}")
    print("=" * 78)

    summaries: list[GranularEvalSummary] = []
    for step in args.steps:
        policy_path = train_dir / f"step_{step:06d}"
        if not policy_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {policy_path}")

        print(f"\n>>> 评估 step_{step:06d} ...")
        summary = evaluate_checkpoint(
            policy_path,
            eval_scenes,
            step=step,
            dataset_root=dataset_root,
            device_name=args.device,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        summaries.append(summary)
        print(f"    完成，耗时 {summary.elapsed_s:.1f}s")

    print_summary_table(summaries)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": args.episodes,
            "seed": args.seed,
            "scene_source": scene_source_label,
            "max_steps": args.max_steps,
            "milestones": MILESTONE_LABELS,
            "checkpoints": [
                {
                    "step": s.step,
                    "counts": s.counts,
                    "rates": s.rates,
                    "elapsed_s": s.elapsed_s,
                    "per_episode": s.per_episode,
                }
                for s in summaries
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\n结果已写入: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
