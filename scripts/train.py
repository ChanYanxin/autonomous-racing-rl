from autonomous_racing_rl.config import TrainingConfig
from autonomous_racing_rl.envs import make_racing_env
from autonomous_racing_rl.training import train


def main() -> None:
    config = TrainingConfig()
    env = make_racing_env(config.seed, render=False)
    eval_env = make_racing_env(config.seed, render=False)
    try:
        train(env, eval_env, config)
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
