from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .agent import Actor, Critic, preprocess_observation
from .buffer import ReplayBuffer
from .config import TrainingConfig
from .evaluation import evaluate_policy


def _soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def train(env, eval_env, config: TrainingConfig, checkpoint_dir: str | Path = "checkpoints") -> Actor:
    """Train a DDPG agent on a compatible continuous-control Gymnasium environment."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_obs, _ = env.reset(seed=config.seed)
    obs = preprocess_observation(raw_obs).to(device)

    observation_dim = int(obs.numel())
    action_dim = int(np.prod(env.action_space.shape))
    actor = Actor(observation_dim, env.action_space.low, env.action_space.high).to(device)
    critic = Critic(observation_dim, action_dim).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.buffer_size, observation_dim, action_dim, device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_eval_return = -np.inf

    for step in range(config.total_timesteps):
        if step < config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(obs).cpu().numpy()
            noise_std = np.asarray(env.action_space.high) * config.exploration_noise
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high)

        raw_next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = preprocess_observation(raw_next_obs).to(device)
        done = bool(terminated)

        replay.add(
            obs.detach().cpu().numpy(),
            np.asarray(action, dtype=np.float32),
            float(reward),
            next_obs.detach().cpu().numpy(),
            done,
        )

        if terminated or truncated:
            raw_reset_obs, _ = env.reset()
            next_obs = preprocess_observation(raw_reset_obs).to(device)
        obs = next_obs

        if step >= config.learning_starts and replay.size >= config.batch_size:
            batch = replay.sample(config.batch_size)
            with torch.no_grad():
                next_actions = target_actor(batch.next_observations)
                target_q = target_critic(batch.next_observations, next_actions)
                td_target = batch.rewards + (1.0 - batch.dones) * config.gamma * target_q

            critic_loss = F.mse_loss(critic(batch.observations, batch.actions), td_target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

            if step % config.policy_frequency == 0:
                actor_loss = -critic(batch.observations, actor(batch.observations)).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()
                _soft_update(actor, target_actor, config.tau)
                _soft_update(critic, target_critic, config.tau)

        if step >= config.learning_starts and step % config.eval_frequency == 0:
            mean_return, mean_length = evaluate_policy(
                eval_env, actor, config.eval_episodes, device
            )
            print(
                f"step={step} mean_eval_return={mean_return:.3f} "
                f"mean_eval_length={mean_length:.1f}"
            )
            torch.save(actor.state_dict(), checkpoint_dir / "last_actor.pt")
            if mean_return > best_eval_return:
                best_eval_return = mean_return
                torch.save(actor.state_dict(), checkpoint_dir / "best_actor.pt")

    return actor
