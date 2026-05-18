from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    name: str
    max_steps: int = 300
    solved_threshold: float = 0.25


# MuJoCo-style task names we validate with MPC controllers.
ENV_REGISTRY: list[EnvSpec] = [
    EnvSpec("inverted_pendulum", 250, 0.09),
    EnvSpec("ant", 300, 0.3),
    EnvSpec("humanoid", 300, 0.31),
]
