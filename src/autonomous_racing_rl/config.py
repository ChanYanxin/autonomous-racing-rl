from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters matching the final racing experiment."""

    seed: int = 1
    total_timesteps: int = 4_000_000
    learning_rate: float = 3e-4
    buffer_size: int = 4_000_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 32
    exploration_noise: float = 0.5
    learning_starts: int = 10_000
    policy_frequency: int = 2
    eval_frequency: int = 10_000
    eval_episodes: int = 1
