#!/usr/bin/env python3
"""PPO 训练抓杯 RL 策略（proprio + delta 动作；可选 ACT residual-v2）。"""

from __future__ import annotations

import argparse
from pathlib import Path

# macOS: 必须先 import torch，再 import numpy（或会间接 import numpy 的库），
# 否则 ACTPolicy.from_pretrained 可能 segfault。
import torch
import numpy as np

from _bootstrap import bootstrap

PICKTASK_DIR = bootstrap(include_lerobot=False)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def make_env(
    mode: str,
    *,
    seed: int,
    randomize: bool,
    act_policy: Path | None,
    act_dataset: Path | None,
    device: torch.device,
    act_bundle=None,
    cup_variant: str | None = None,
):
    # macOS residual: 必须先加载 ACT（from_pretrained），再 import mujoco / gym env。
    if mode == "residual":
        if act_policy is None or act_dataset is None:
            raise ValueError("residual 模式需要 --act-policy 与 --act-dataset")
        bootstrap(include_lerobot=True)
        if act_bundle is None:
            from pickcup_act_inference import load_act_inference_bundle

            act_bundle = load_act_inference_bundle(act_policy, act_dataset, device)
        from pickcup_gym_env import PickCupEnvConfig, ResidualActPickCupEnv

        cfg = PickCupEnvConfig(randomize_scene=randomize, cup_variant=cup_variant)
        return ResidualActPickCupEnv(act_bundle, config=cfg, seed=seed)

    from pickcup_gym_env import PickCupEnv, PickCupEnvConfig

    cfg = PickCupEnvConfig(randomize_scene=randomize, cup_variant=cup_variant)
    return PickCupEnv(config=cfg, seed=seed)


def train(
    *,
    mode: str,
    output_dir: Path,
    timesteps: int,
    seed: int,
    randomize: bool,
    act_policy: Path | None,
    act_dataset: Path | None,
    device_name: str,
    learning_rate: float,
    eval_every: int,
    eval_episodes: int,
    cup_variant: str | None = None,
) -> Path:
    device = pick_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # macOS: ACT from_pretrained 必须在 import gymnasium / mujoco 之前。
    shared_act_bundle = None
    if mode == "residual":
        if act_policy is None or act_dataset is None:
            raise ValueError("residual 模式需要 --act-policy 与 --act-dataset")
        bootstrap(include_lerobot=True)
        from pickcup_act_inference import load_act_inference_bundle

        print(f"加载 ACT: {act_policy}", flush=True)
        shared_act_bundle = load_act_inference_bundle(act_policy, act_dataset, device)
        print("ACT 加载完成", flush=True)

    from pickcup_ppo import PPOAgent, evaluate_success_rate

    train_env = make_env(
        mode,
        seed=seed,
        randomize=randomize,
        act_policy=act_policy,
        act_dataset=act_dataset,
        device=device,
        act_bundle=shared_act_bundle,
        cup_variant=cup_variant,
    )
    eval_env = make_env(
        mode,
        seed=seed + 10_000,
        randomize=randomize,
        act_policy=act_policy,
        act_dataset=act_dataset,
        device=device,
        act_bundle=shared_act_bundle,
        cup_variant=cup_variant,
    )

    agent = PPOAgent.from_env(
        train_env,
        lr=learning_rate,
        device=torch.device("cpu"),
        residual=(mode == "residual"),
    )
    rollout_steps = agent.cfg.rollout_steps
    total_updates = max(timesteps // rollout_steps, 1)
    log_every = 2 if mode == "residual" else 10

    print("=" * 60)
    print("Pick Cup RL 训练 (PPO / PyTorch)")
    print("=" * 60)
    print(f"  mode:         {mode}")
    print(f"  timesteps:    {timesteps}")
    print(f"  updates:      {total_updates}")
    print(f"  eval_every:   {eval_every}")
    print(f"  randomize:    {randomize}")
    print(f"  cup:          {getattr(train_env, 'cup_variant', cup_variant)}")
    print(f"  output:       {output_dir}")
    print(f"  obs_dim:      {agent.cfg.obs_dim}")
    print(f"  log_std_init: {agent.cfg.log_std_init}")
    print(f"  ent_coef:     {agent.cfg.ent_coef}")
    if mode == "residual":
        from pickcup_gym_env import DEFAULT_RESIDUAL_SCALES
        from pickcup_reward import RewardConfig

        print(f"  act_policy:   {act_policy}")
        print(f"  act_dataset:  {act_dataset}")
        print(f"  residual_scales: {DEFAULT_RESIDUAL_SCALES.tolist()}")
        print(f"  reward:       {RewardConfig.for_residual()}")
        print("  preset:       residual-v2 (small Δ, lock gripper, hold-aware reward)")
    print("=" * 60)

    best_rate = -1.0
    best_path = output_dir / "ppo_pickcup_best.pt"

    for update in range(1, total_updates + 1):
        rollout_seed = int(rng.integers(0, 2**31 - 1))
        batch = agent.collect_rollout(train_env, seed=rollout_seed)
        stats = agent.update(batch)
        mean_reward = float(batch.rewards.mean())

        if update % max(eval_every // rollout_steps, 1) == 0 or update == total_updates:
            rate, _ = evaluate_success_rate(agent, eval_env, episodes=eval_episodes, seed=seed + update)
            print(
                f"[{update:4d}/{total_updates}] "
                f"reward={mean_reward:7.3f} "
                f"pi_loss={stats['policy_loss']:.4f} "
                f"vf_loss={stats['value_loss']:.4f} "
                f"kl={stats['approx_kl']:.4f} "
                f"skip={stats['skipped']:.1f} "
                f"eval={rate * 100:.1f}%",
                flush=True,
            )
            if rate > best_rate:
                best_rate = rate
                agent.save(best_path)
        elif update % log_every == 0:
            print(
                f"[{update:4d}/{total_updates}] reward={mean_reward:7.3f} "
                f"pi_loss={stats['policy_loss']:.4f} vf_loss={stats['value_loss']:.4f} "
                f"kl={stats['approx_kl']:.4f} skip={stats['skipped']:.1f}",
                flush=True,
            )

        ckpt_every = max((2_000 if mode == "residual" else 10_000) // rollout_steps, 1)
        if update % ckpt_every == 0:
            ckpt = output_dir / "checkpoints" / f"ppo_pickcup_{update:06d}.pt"
            agent.save(ckpt)

    final_path = output_dir / "ppo_pickcup_final.pt"
    agent.save(final_path)
    print(f"\n已保存 RL 策略 -> {final_path}")
    if best_path.exists():
        print(f"最佳 eval 策略 -> {best_path} ({best_rate * 100:.1f}%)")

    train_env.close()
    eval_env.close()
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 训练 pickcup RL / residual RL")
    parser.add_argument(
        "--mode",
        choices=["ppo", "residual"],
        default="ppo",
        help="ppo=纯 delta RL；residual=ACT + 小残差（v2）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认：ppo→outputs/rl/ppo_pickcup；residual→outputs/rl/residual_act_v2",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="默认：ppo=200000；residual=30000",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-randomize", action="store_true", help="固定默认场景")
    parser.add_argument(
        "--act-policy",
        type=Path,
        default=None,
        help="residual 模式：ACT checkpoint 目录",
    )
    parser.add_argument(
        "--act-dataset",
        type=Path,
        default=PICKTASK_DIR / "pickcupdata" / "train" / "pickcup_train_200",
        help="residual 模式：ACT 训练 dataset（stats）",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="默认：ppo=10000；residual=2000",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="默认：ppo=5；residual=8",
    )
    parser.add_argument(
        "--cup",
        choices=("yellow", "red"),
        default=None,
        help="杯子变体（默认 yellow；也可用环境变量 PICKCUP_CUP_VARIANT）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    timesteps = args.timesteps if args.timesteps is not None else (30_000 if mode == "residual" else 200_000)
    eval_every = args.eval_every if args.eval_every is not None else (2_000 if mode == "residual" else 10_000)
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else (8 if mode == "residual" else 5)
    output = args.output
    if output is None:
        output = (
            PICKTASK_DIR.parent / "outputs" / "rl" / "residual_act_v2"
            if mode == "residual"
            else PICKTASK_DIR.parent / "outputs" / "rl" / "ppo_pickcup"
        )

    train(
        mode=mode,
        output_dir=output.resolve(),
        timesteps=timesteps,
        seed=args.seed,
        randomize=not args.no_randomize,
        act_policy=args.act_policy.resolve() if args.act_policy else None,
        act_dataset=args.act_dataset.resolve() if args.act_dataset else None,
        device_name=args.device,
        learning_rate=args.lr,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        cup_variant=args.cup,
    )


if __name__ == "__main__":
    main()
