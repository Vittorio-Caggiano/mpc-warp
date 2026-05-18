"""Fork-based environment wrapper used only by the synthetic-env test stack."""
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

    def get_state(self) -> dict:
        if hasattr(self.env, "get_internal_state"):
            return self.env.get_internal_state()
        raise NotImplementedError("Environment does not support get_internal_state")

    def set_state(self, snapshot: dict) -> list[float]:
        if hasattr(self.env, "set_internal_state"):
            obs = self.env.set_internal_state(snapshot)
            return [float(x) for x in obs]
        raise NotImplementedError("Environment does not support set_internal_state")

    def fork_with_state(self, snapshot: dict) -> "MujocoWarpBackend":
        forked = MujocoWarpBackend(self._env_factory)
        forked.set_state(snapshot)
        return forked

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
