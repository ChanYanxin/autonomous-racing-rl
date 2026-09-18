import numpy as np
import pytest
import torch

from autonomous_racing_rl.buffer import ReplayBuffer


def test_replay_buffer_add_and_sample() -> None:
    buffer = ReplayBuffer(10, observation_dim=4, action_dim=2, device=torch.device("cpu"))
    for i in range(6):
        buffer.add(
            np.full(4, i, dtype=np.float32),
            np.full(2, i, dtype=np.float32),
            float(i),
            np.full(4, i + 1, dtype=np.float32),
            done=i % 2 == 0,
        )
    batch = buffer.sample(4)
    assert batch.observations.shape == (4, 4)
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4, 1)


def test_replay_buffer_rejects_undersized_sample() -> None:
    buffer = ReplayBuffer(4, 2, 1, torch.device("cpu"))
    with pytest.raises(ValueError):
        buffer.sample(1)
