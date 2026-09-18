import argparse

import numpy as np
import torch

from autonomous_racing_rl.agent import Actor, preprocess_observation
from autonomous_racing_rl.envs import make_racing_env
from autonomous_racing_rl.evaluation import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to an actor state_dict (.pt)")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = make_racing_env(seed=1, render=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        raw_obs, _ = env.reset(seed=1)
        observation_dim = int(preprocess_observation(raw_obs).numel())
        actor = Actor(
            observation_dim,
            np.asarray(env.action_space.low),
            np.asarray(env.action_space.high),
        ).to(device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
        mean_return, mean_length = evaluate_policy(env, actor, args.episodes, device)
        print(f"mean_return={mean_return:.4f} mean_episode_length={mean_length:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
