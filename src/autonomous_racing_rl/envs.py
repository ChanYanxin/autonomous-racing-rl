from __future__ import annotations

from typing import Any

import gymnasium as gym


def make_racing_env(seed: int, render: bool = False, render_width: int = 1280) -> gym.Env:
    """Create the original course racing environment when it is installed.

    `CarEnv` is intentionally not distributed with this portfolio repository.
    """
    try:
        import CarEnv  # noqa: F401  # type: ignore
        from CarEnv.Configs import RACING_FAST  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CarEnv is not installed. Obtain the RLLBC BPA3 environment through "
            "the authorized course distribution and install it separately."
        ) from exc

    kwargs: dict[str, Any] = {
        "config": RACING_FAST,
        "render_env": render,
        "limit_speed_factor": None,
        "render_width": render_width,
    }
    env = gym.make("CarEnv:gym_envs/CarEnv-v1", **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=6000)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
