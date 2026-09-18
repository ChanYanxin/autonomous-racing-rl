# Autonomous Racing with Deep Reinforcement Learning

A portfolio-oriented, clean-room reconstruction of my solution for the **RWTH Aachen Reinforcement Learning & Learning-based Control (RLLBC) BPA3 Car Racing Challenge**.

The task is continuous-control autonomous racing: an agent receives a high-dimensional observation of vehicle state, track geometry and perceived cones, and outputs **steering, throttle and brake** commands. The objective is to drive quickly while avoiding collisions and leaving the track.

> **Official evaluation:** mean return **93.2585** across 10 evaluation episodes (600-step cap per episode) for the submitted policy.

## What I implemented

- DDPG-style actor-critic training for a 3D continuous action space.
- Observation preprocessing for the 456-dimensional simulator observation, removing non-policy RGB payload entries before inference.
- Deterministic actor network with action rescaling to the environment bounds.
- Replay-buffer training, target networks and Polyak updates.
- Gaussian exploration noise for data collection.
- Periodic deterministic evaluation and best-checkpoint selection.
- Reproducible configuration, unit tests and CI for the public reconstruction.

## Repository scope

This repository intentionally **does not redistribute the course simulator (`CarEnv`), starter repository, task sheets, or private model artifacts**. The public code is a clean-room reconstruction of the learning pipeline and project structure for portfolio use.

To run against the original racing environment, you need legitimate access to the RLLBC BPA3 environment and must install it separately. The environment adapter uses a lazy import, so the package and tests remain usable without the course simulator.

## Project structure

```text
.
├── src/autonomous_racing_rl/
│   ├── agent.py            # actor, critic and observation preprocessing
│   ├── buffer.py           # replay buffer
│   ├── config.py           # experiment configuration
│   ├── envs.py             # optional CarEnv adapter
│   ├── evaluation.py       # deterministic policy evaluation
│   └── training.py         # DDPG training loop
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── official_evaluation.md
├── docs/
│   └── architecture.md
├── tests/
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Algorithm

The actor maps the preprocessed observation to three bounded continuous controls. The critic estimates \(Q(s,a)\). Training alternates between:

1. collecting transitions with exploration noise,
2. minimizing the critic TD error,
3. maximizing the critic value of the actor's action,
4. slowly updating actor and critic target networks.

The public reconstruction keeps the architecture close to the submitted solution: two 64-unit hidden layers for the actor and critic, ReLU activations, and a tanh actor output rescaled to the action limits.

## Final training configuration

| Parameter | Value |
|---|---:|
| Total environment steps | 4,000,000 |
| Learning rate | 3e-4 |
| Replay buffer capacity | 4,000,000 |
| Discount factor | 0.99 |
| Target update coefficient | 0.005 |
| Batch size | 32 |
| Exploration noise scale | 0.5 |
| Learning starts | 10,000 steps |
| Actor update frequency | every 2 critic steps |
| Evaluation frequency | every 10,000 steps |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

For the original course environment, install the provided `CarEnv` package separately, then run:

```bash
python scripts/train.py
```

The public repository cannot provide that dependency.

## Testing

```bash
pytest
```

The tests use small synthetic Gymnasium spaces and do not require `CarEnv`.

## Evaluation result

The submitted racing policy achieved a **mean return of 93.2585** in the official 10-episode evaluation. See [`results/official_evaluation.md`](results/official_evaluation.md) for provenance and limitations.

## Skills demonstrated

`Deep Reinforcement Learning` · `DDPG` · `PyTorch` · `Gymnasium` · `Continuous Control` · `Autonomous Racing` · `Experiment Design` · `Evaluation` · `Reproducible ML Engineering`

## Acknowledgements

The original racing environment and assignment were provided in the RWTH Aachen RLLBC course. This repository contains only my public clean-room reconstruction of the learning pipeline and documentation; course-owned materials are excluded.
