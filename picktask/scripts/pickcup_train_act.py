#!/usr/bin/env python3
"""Mac 友好的 ACT 训练脚本（本地 dataset，无需完整 lerobot-train 依赖链）。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from _bootstrap import bootstrap

PICKTASK_DIR = bootstrap(include_lerobot=True)
WORKSPACE_DIR = PICKTASK_DIR.parent

from lerobot_video_patch import apply_lerobot_video_patch

apply_lerobot_video_patch()

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def save_checkpoint(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)

    config_path = checkpoint_dir / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict["type"] = "act"
    config_path.write_text(json.dumps(config_dict, indent=4) + "\n")


def format_loss_log(step: int, loss: float, loss_dict: dict) -> str:
    parts = [f"step {step:5d}", f"loss={loss:.4f}"]
    if "l1_loss" in loss_dict:
        parts.append(f"l1={loss_dict['l1_loss']:.4f}")
    if "kld_loss" in loss_dict:
        parts.append(f"kld={loss_dict['kld_loss']:.4f}")
    return "  ".join(parts)


def infer_step_from_checkpoint(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)", path.name)
    if match is None:
        return 0
    return int(match.group(1))


def train(
    dataset_root: Path,
    output_dir: Path,
    *,
    repo_id: str,
    steps: int,
    batch_size: int,
    log_freq: int,
    save_freq: int,
    device_name: str,
    resume_policy: Path | None,
    start_step: int | None,
) -> Path:
    device = pick_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if resume_policy is None:
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            device=device.type,
        )
        policy = ACTPolicy(cfg)
        initial_step = 0
    else:
        policy = ACTPolicy.from_pretrained(resume_policy)
        cfg = policy.config
        initial_step = start_step if start_step is not None else infer_step_from_checkpoint(resume_policy)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=metadata.stats)
    policy.train()
    policy.to(device)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, metadata.fps),
    }
    delta_timestamps |= {
        key: make_delta_timestamps(cfg.observation_delta_indices, metadata.fps)
        for key in cfg.image_features
    }

    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
        num_workers=0,
    )

    print("=" * 60)
    print("ACT 训练 (pickcup)")
    print("=" * 60)
    print(f"  dataset: {dataset_root}")
    print(f"  episodes: {metadata.total_episodes}, frames: {metadata.total_frames}")
    print(f"  device: {device}, batch_size: {batch_size}, steps: {steps}")
    if resume_policy is not None:
        print(f"  resume: {resume_policy} (start_step={initial_step}, optimizer=fresh)")
    print(f"  log_freq: {log_freq}, save_freq: {save_freq}")
    print(f"  video decode: pyav patch (torchvision 无 VideoReader)")
    print(f"  output: {output_dir}")
    print(f"  checkpoints: {output_dir / 'checkpoints'}")
    print("=" * 60)

    checkpoint_root = output_dir / "checkpoints"
    step = initial_step
    while step < steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(format_loss_log(step, loss.item(), loss_dict))

            step += 1
            if save_freq > 0 and step % save_freq == 0:
                ckpt_dir = checkpoint_root / f"step_{step:06d}"
                save_checkpoint(policy, preprocessor, postprocessor, ckpt_dir)
                print(f"  -> checkpoint saved: {ckpt_dir}")

            if step >= steps:
                break

    save_checkpoint(policy, preprocessor, postprocessor, output_dir)

    print(f"\n已保存 policy -> {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 pickcup ACT 策略")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PICKTASK_DIR / "pickcupdata" / "train" / "pickcup_train_mac",
        help="LeRobot dataset 根目录",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/pickcup_train_mac",
        help="dataset repo_id（与目录名对应即可）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_DIR / "outputs" / "train" / "act_pickcup",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="每 N step 保存 checkpoint 到 output/checkpoints/step_XXXXXX（0=关闭）",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--resume-policy",
        type=Path,
        default=None,
        help="从已有 ACT checkpoint 目录继续训练（例如 checkpoints/step_030000）",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="续训起始 step；默认从 resume-policy 目录名 step_XXXXXX 自动推断",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        args.dataset.resolve(),
        args.output.resolve(),
        repo_id=args.repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        device_name=args.device,
        resume_policy=args.resume_policy.resolve() if args.resume_policy else None,
        start_step=args.start_step,
    )


if __name__ == "__main__":
    main()
