import torch
from torch import nn

from autonomous_racing_rl.training import _soft_update


def test_soft_update() -> None:
    source = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(1.0)
        target.weight.fill_(0.0)
    _soft_update(source, target, tau=0.25)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.25))
