from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class StepResult:
    state: list[float]
    reward: float = 0.0
    done: bool = False


class MujocoWarpBackend:
    def __init__(self, env_factory: Callable[[], object]):
        self._env_factory = env_factory
        self.env = env_factory()

    def reset(self, seed: int | None = None) -> list[float]:
        out = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return [float(x) for x in obs]

    def step(self, action: list[float]) -> StepResult:
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, _info = out
            done = bool(terminated or truncated)
        elif len(out) == 4:
            obs, reward, done, _info = out
        else:
            raise ValueError(f"Unexpected step return length: {len(out)}")
        return StepResult(state=[float(x) for x in obs], reward=float(reward), done=bool(done))

    def clone(self) -> "MujocoWarpBackend":
        return MujocoWarpBackend(self._env_factory)
