"""轻量 PPO 实现（纯 PyTorch，无需 stable-baselines3）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    lr: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01  # residual 建议 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.25
    rollout_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 3
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    max_ratio: float = 10.0
    target_kl: float = 0.01
    obs_clip: float = 10.0
    mean_clip: float = 5.0
    value_clip: float = 100.0
    # residual-v2：更小探索，从近零残差起步
    zero_init_policy: bool = False


class ActorCritic(nn.Module):
    def __init__(self, cfg: PPOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        last = cfg.obs_dim
        for width in cfg.hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.Tanh()])
            last = width
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, cfg.action_dim)
        self.value_head = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.log_std_init))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=self.cfg.obs_clip, neginf=-self.cfg.obs_clip)
        obs = torch.clamp(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        features = self.backbone(obs)
        features = torch.nan_to_num(features, nan=0.0, posinf=self.cfg.mean_clip, neginf=-self.cfg.mean_clip)
        mean = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=self.cfg.mean_clip, neginf=-self.cfg.mean_clip)
        value = torch.nan_to_num(value, nan=0.0, posinf=self.cfg.value_clip, neginf=-self.cfg.value_clip)
        return torch.clamp(mean, -self.cfg.mean_clip, self.cfg.mean_clip), torch.clamp(
            value,
            -self.cfg.value_clip,
            self.cfg.value_clip,
        )

    def _distribution(self, mean: torch.Tensor) -> Normal:
        mean = torch.nan_to_num(mean, nan=0.0, posinf=self.cfg.mean_clip, neginf=-self.cfg.mean_clip)
        mean = torch.clamp(mean, -self.cfg.mean_clip, self.cfg.mean_clip)
        log_std = torch.nan_to_num(self.log_std, nan=self.cfg.log_std_init, posinf=self.cfg.log_std_max, neginf=self.cfg.log_std_min)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std, validate_args=False)

    @staticmethod
    def _atanh(action: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(action) - torch.log1p(-action))

    @staticmethod
    def _squashed_log_prob(dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Tanh-Gaussian correction keeps bounded actions and log-probs consistent.
        eps = 1e-6
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + eps)
        return log_prob.sum(-1)

    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        dist = self._distribution(mean)
        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        dist = self._distribution(mean)
        raw_actions = self._atanh(actions)
        log_prob = self._squashed_log_prob(dist, raw_actions, actions)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


@dataclass
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class PPOAgent:
    def __init__(self, cfg: PPOConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(cfg).to(device)
        if cfg.zero_init_policy:
            nn.init.zeros_(self.net.policy_head.weight)
            nn.init.zeros_(self.net.policy_head.bias)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

    @classmethod
    def from_env(
        cls,
        env: gym.Env,
        *,
        lr: float = 5e-5,
        device: torch.device | None = None,
        residual: bool = False,
    ) -> PPOAgent:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        if residual:
            cfg = PPOConfig(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=lr,
                log_std_init=-2.5,
                ent_coef=0.001,
                zero_init_policy=True,
            )
        else:
            cfg = PPOConfig(obs_dim=obs_dim, action_dim=action_dim, lr=lr)
        return cls(cfg, device or torch.device("cpu"))

    def _sanitize_parameters(self) -> None:
        with torch.no_grad():
            self.net.log_std.clamp_(self.cfg.log_std_min, self.cfg.log_std_max)
            for name, param in self.net.named_parameters():
                if torch.isfinite(param).all():
                    continue
                fill = self.cfg.log_std_init if name == "log_std" else 0.0
                param.nan_to_num_(nan=fill, posinf=1.0, neginf=-1.0)

    def _gradients_are_finite(self) -> bool:
        for param in self.net.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
        return True

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.cfg.obs_clip, neginf=-self.cfg.obs_clip)
        obs = np.clip(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def predict(self, obs: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        self._sanitize_parameters()
        obs_t = self._obs_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.net.act(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def collect_rollout(self, env: gym.Env, *, seed: int | None = None) -> RolloutBatch:
        cfg = self.cfg
        self._sanitize_parameters()
        obs_buf = np.zeros((cfg.rollout_steps, cfg.obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, cfg.action_dim), dtype=np.float32)
        logp_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)

        obs, _ = env.reset(seed=seed)
        if hasattr(env, "act_bundle"):
            env.act_bundle.reset()

        for step in range(cfg.rollout_steps):
            obs_buf[step] = np.nan_to_num(
                obs,
                nan=0.0,
                posinf=cfg.obs_clip,
                neginf=-cfg.obs_clip,
            )
            obs_buf[step] = np.clip(obs_buf[step], -cfg.obs_clip, cfg.obs_clip)
            obs_t = self._obs_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.net.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            act_buf[step] = action_np
            logp_buf[step] = float(torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0).item())
            rew_buf[step] = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
            done_buf[step] = float(done)
            val_buf[step] = float(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).item())

            if done:
                obs, _ = env.reset()
                if hasattr(env, "act_bundle"):
                    env.act_bundle.reset()

        with torch.no_grad():
            obs_t = self._obs_tensor(obs).unsqueeze(0)
            _, last_value = self.net(obs_t)
            last_value = float(torch.nan_to_num(last_value, nan=0.0, posinf=0.0, neginf=0.0).item())

        advantages = np.zeros(cfg.rollout_steps, dtype=np.float32)
        last_gae = 0.0
        for step in reversed(range(cfg.rollout_steps)):
            if step == cfg.rollout_steps - 1:
                next_non_terminal = 1.0 - done_buf[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - done_buf[step]
                next_value = val_buf[step + 1]
            delta = rew_buf[step] + cfg.gamma * next_value * next_non_terminal - val_buf[step]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + val_buf

        return RolloutBatch(
            obs=obs_buf,
            actions=act_buf,
            log_probs=logp_buf,
            rewards=rew_buf,
            dones=done_buf,
            values=val_buf,
            advantages=advantages,
            returns=returns,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        cfg = self.cfg
        self._sanitize_parameters()
        obs = torch.as_tensor(batch.obs, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, device=self.device)
        advantages = torch.as_tensor(batch.advantages, device=self.device)
        returns = torch.as_tensor(batch.returns, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        n = cfg.rollout_steps
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "skipped": 0.0}
        updates = 0
        stop_update = False

        for _ in range(cfg.n_epochs):
            if stop_update:
                break
            np.random.shuffle(indices)
            for start in range(0, n, cfg.batch_size):
                end = start + cfg.batch_size
                mb = indices[start:end]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_logp = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]

                new_logp, entropy, values = self.net.evaluate(mb_obs, mb_actions)
                log_ratio = torch.clamp(new_logp - mb_old_logp, -20.0, 20.0)
                ratio = torch.exp(log_ratio).clamp(max=cfg.max_ratio)
                approx_kl = (mb_old_logp - new_logp).mean()
                if not torch.isfinite(log_ratio).all() or not torch.isfinite(values).all():
                    stats["skipped"] += 1.0
                    continue
                if approx_kl.item() > cfg.target_kl * 1.5:
                    stop_update = True
                    break
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss
                if not torch.isfinite(loss):
                    stats["skipped"] += 1.0
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                if not self._gradients_are_finite():
                    self.optimizer.zero_grad(set_to_none=True)
                    stats["skipped"] += 1.0
                    continue
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                if not self._gradients_are_finite():
                    self.optimizer.zero_grad(set_to_none=True)
                    stats["skipped"] += 1.0
                    continue
                self.optimizer.step()
                self._sanitize_parameters()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(-entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                updates += 1

        for key in stats:
            stats[key] /= max(updates, 1)
        return stats

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "model_state_dict": self.net.state_dict(),
        }
        torch.save(payload, path)
        config_path = path.with_suffix(".json")
        config_path.write_text(json.dumps(payload["config"], indent=2) + "\n")

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> PPOAgent:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = PPOConfig(**payload["config"])
        agent = cls(cfg, device or torch.device("cpu"))
        agent.net.load_state_dict(payload["model_state_dict"])
        agent.net.eval()
        return agent


def evaluate_success_rate(
    agent: PPOAgent,
    env: gym.Env,
    *,
    episodes: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    successes = 0
    records: list[dict[str, float]] = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        if hasattr(env, "act_bundle"):
            env.act_bundle.reset()
        done = False
        truncated = False
        total_reward = 0.0
        info: dict = {}
        while not (done or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        success = bool(info.get("success", False))
        successes += int(success)
        records.append(
            {
                "success": float(success),
                "max_lift": float(info.get("max_lift", 0.0)),
                "reward": total_reward,
            }
        )
    return successes / max(episodes, 1), records
