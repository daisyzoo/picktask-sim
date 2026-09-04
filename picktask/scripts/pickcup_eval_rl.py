#!/usr/bin/env python3
"""评估 PPO / residual RL 策略在 MuJoCo 抓杯任务上的成功率。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# macOS: 必须先 import torch，再 import numpy，否则 ACT from_pretrained 可能 segfault。
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


def evaluate(
    policy_path: Path,
    *,
    mode: str,
    episodes: int,
    seed: int,
    randomize: bool,
    act_policy: Path | None,
    act_dataset: Path | None,
    device_name: str,
    render: bool,
    cup_variant: str | None = None,
) -> float:
    device = pick_device(device_name)
    rng = np.random.default_rng(seed)
    successes = 0

    # macOS: ACT from_pretrained 必须在 import gymnasium（via pickcup_ppo）之前。
    shared_act_bundle = None
    if mode == "residual":
        if act_policy is None or act_dataset is None:
            raise ValueError("residual 模式需要 --act-policy 与 --act-dataset")
        bootstrap(include_lerobot=True)
        from pickcup_act_inference import load_act_inference_bundle

        print(f"加载 ACT: {act_policy}", flush=True)
        shared_act_bundle = load_act_inference_bundle(act_policy, act_dataset, device)
        print("ACT 加载完成", flush=True)

    from pickcup_ppo import PPOAgent

    agent = PPOAgent.load(policy_path, device=torch.device("cpu"))

    print("=" * 60)
    print("MuJoCo RL 闭环评估")
    print("=" * 60)
    print(f"  policy:   {policy_path}")
    print(f"  mode:     {mode}")
    print(f"  cup:      {cup_variant or 'yellow(default)'}")
    print(f"  episodes: {episodes}")
    print("=" * 60)

    for index in range(episodes):
        env = make_env(
            mode,
            seed=int(rng.integers(0, 2**31 - 1)),
            randomize=randomize,
            act_policy=act_policy,
            act_dataset=act_dataset,
            device=device,
            act_bundle=shared_act_bundle,
            cup_variant=cup_variant,
        )
        obs, _ = env.reset()
        if mode == "residual" and hasattr(env, "act_bundle"):
            env.act_bundle.reset()

        done = False
        truncated = False
        total_reward = 0.0
        t0 = time.time()
        info: dict = {}

        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if render:
                frame = env.render()
                if frame is not None:
                    try:
                        import cv2

                        cv2.imshow("pickcup_rl_eval", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    except Exception:
                        pass

        elapsed = time.time() - t0
        success = bool(info.get("success", False))
        if success:
            successes += 1
        status = "OK" if success else "FAIL"
        print(
            f"[{index + 1:3d}/{episodes}] {status} "
            f"lift={info.get('max_lift', 0.0):.3f}m reward={total_reward:.2f} t={elapsed:.1f}s"
        )
        env.close()

    rate = successes / max(episodes, 1)
    print(f"\n成功率: {successes}/{episodes} = {rate * 100:.1f}%")
    return rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 pickcup PPO / residual RL 策略")
    parser.add_argument("--policy", type=Path, required=True, help="PPO checkpoint (.pt)")
    parser.add_argument("--mode", choices=["ppo", "residual"], default="ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--act-policy", type=Path, default=None)
    parser.add_argument(
        "--act-dataset",
        type=Path,
        default=PICKTASK_DIR / "pickcupdata" / "train" / "pickcup_train_200",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--cup",
        choices=("yellow", "red"),
        default=None,
        help="杯子变体（默认 yellow；也可用环境变量 PICKCUP_CUP_VARIANT）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rate = evaluate(
        args.policy.resolve(),
        mode=args.mode,
        episodes=args.episodes,
        seed=args.seed,
        randomize=not args.no_randomize,
        act_policy=args.act_policy.resolve() if args.act_policy else None,
        act_dataset=args.act_dataset.resolve() if args.act_dataset else None,
        device_name=args.device,
        render=args.render,
        cup_variant=args.cup,
    )
    raise SystemExit(0 if rate > 0.0 else 1)


if __name__ == "__main__":
    main()
