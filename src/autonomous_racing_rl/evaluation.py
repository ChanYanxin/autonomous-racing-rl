from __future__ import annotations

import numpy as np
import torch

from .agent import Actor, preprocess_observation


def evaluate_policy(env, actor: Actor, episodes: int, device: torch.device) -> tuple[float, float]:
    """Run deterministic policy evaluation and return mean return and episode length."""
    returns: list[float] = []
    lengths: list[int] = []

    actor.eval()
    for episode in range(episodes):
        raw_obs, _ = env.reset(seed=episode)
        obs = preprocess_observation(raw_obs).to(device)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                action = actor(obs).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            raw_obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess_observation(raw_obs).to(device)
            total_reward += float(reward)
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)

    return float(np.mean(returns)), float(np.mean(lengths))
