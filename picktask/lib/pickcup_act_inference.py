"""加载 ACT 策略用于 RL residual 模式（与 pickcup_sim_eval 逻辑一致，独立模块）。

macOS 注意：
1. 必须先 import torch，再 import numpy（或 gymnasium 等会拉 numpy 的库），
   否则 ACTPolicy.from_pretrained 可能 segfault。
2. 必须先 ACTPolicy.from_pretrained，再 import mujoco / pickcup /
   LeRobotDatasetMetadata。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# macOS: torch 必须在 numpy 之前导入。
import torch
import numpy as np

# 与 lerobot_recorder.DEFAULT_TASK 保持一致；此处硬编码以避免提前拉入录制依赖。
DEFAULT_TASK = "pick up the cup"


@dataclass
class ActInferenceBundle:
    policy: torch.nn.Module
    metadata: object
    preprocess: object
    postprocess: object
    build_inference_frame: object
    make_robot_action: object
    device: torch.device
    proprio_state_names: tuple[str, ...]
    action_names: tuple[str, ...]

    def predict_action(self, rgb: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        obs_dict = {
            "head_camera": rgb,
            **{
                name: float(proprio[index])
                for index, name in enumerate(self.proprio_state_names)
            },
        }
        obs_frame = self.build_inference_frame(
            observation=obs_dict,
            device=self.device,
            ds_features=self.metadata.features,
            task=DEFAULT_TASK,
        )
        batch = self.preprocess(obs_frame)
        with torch.no_grad():
            action_tensor = self.policy.select_action(batch)
            action_tensor = self.postprocess(action_tensor)
        action_dict = self.make_robot_action(action_tensor, self.metadata.features)
        return np.array(
            [action_dict[name] for name in self.action_names],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.policy.reset()


def load_act_inference_bundle(
    policy_path: Path,
    dataset_root: Path,
    device: torch.device,
) -> ActInferenceBundle:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    config_path = policy_path / "config.json"
    config_dict = json.loads(config_path.read_text())
    if config_dict.get("type") != "act":
        config_dict.setdefault("type", "act")
        config_path.write_text(json.dumps(config_dict, indent=4) + "\n")

    # macOS: 必须先 from_pretrained，再 import LeRobotDatasetMetadata，最后才是 mujoco/pickcup。
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    # checkpoint config 里常是 device=cpu；与实际推理设备对齐，避免 batch 在 CPU、权重在 MPS。
    policy.config.device = device.type
    policy.to(device)

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(
        repo_id=f"local/{dataset_root.name}",
        root=dataset_root,
    )
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )

    import pickcup as pc

    return ActInferenceBundle(
        policy=policy,
        metadata=metadata,
        preprocess=preprocess,
        postprocess=postprocess,
        build_inference_frame=build_inference_frame,
        make_robot_action=make_robot_action,
        device=device,
        proprio_state_names=tuple(pc.PROPRIO_STATE_NAMES),
        action_names=tuple(pc.ACTION_NAMES),
    )
