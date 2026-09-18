import numpy as np
import pytest
import torch

from autonomous_racing_rl.agent import Actor, Critic, preprocess_observation


def test_preprocess_removes_three_entries() -> None:
    observation = np.arange(456, dtype=np.float32)
    processed = preprocess_observation(observation)
    assert processed.shape == (453,)
    assert processed.dtype == torch.float32
    expected = np.delete(observation, [7, 8, 9])
    np.testing.assert_allclose(processed.numpy(), expected)


def test_preprocess_rejects_matrix() -> None:
    with pytest.raises(ValueError):
        preprocess_observation(np.zeros((2, 456), dtype=np.float32))


def test_actor_respects_action_bounds() -> None:
    low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    actor = Actor(453, low, high)
    action = actor(torch.randn(453))
    assert action.shape == (3,)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_critic_output_shape() -> None:
    critic = Critic(453, 3)
    q = critic(torch.randn(8, 453), torch.randn(8, 3))
    assert q.shape == (8, 1)
