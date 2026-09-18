# System architecture

```text
456-D simulator observation
        |
        v
Observation preprocessing
(remove 3 RGB payload entries)
        |
        v
453-D policy state
        |
        +----------------------+
        |                      |
        v                      v
Actor: 453 -> 64 -> 64 -> 3   Critic: (453 + 3) -> 64 -> 64 -> 1
        |                      ^
        v                      |
[steering, throttle, brake] ---+
        |
        v
Racing environment
```

## Training loop

The implementation follows a DDPG-style off-policy actor-critic workflow:

1. Sample an action from the actor and add Gaussian exploration noise.
2. Execute the action in the racing environment.
3. Store the transition in replay memory.
4. Sample mini-batches after an initial warm-up period.
5. Update the critic with a one-step bootstrapped TD target from target networks.
6. Update the actor by maximizing the critic estimate.
7. Apply Polyak averaging to both target networks.
8. Periodically run deterministic evaluation episodes and save the best actor.

## Boundary of the public repository

The racing simulator was supplied by the course. This public repository does not claim authorship of the simulator, vehicle dynamics, track generator, observation specification, or evaluation harness. It documents and reconstructs the learning/control component built around that environment.
