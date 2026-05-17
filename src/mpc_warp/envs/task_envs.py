from __future__ import annotations

import math
import random


class BaseTaskEnv:
    def __init__(self, obs_dim: int, act_dim: int, drift: float, control_gain: float, noise: float):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.drift = drift
        self.control_gain = control_gain
        self.noise = noise
        self.rng = random.Random(0)
        self.obs = [0.0] * obs_dim

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.obs = [self.rng.uniform(-1.0, 1.0) for _ in range(self.obs_dim)]
        return self.obs, {}

    def get_internal_state(self) -> dict:
        return {"obs": list(self.obs), "rng_state": self.rng.getstate()}

    def set_internal_state(self, snapshot: dict):
        self.obs = [float(x) for x in snapshot["obs"]]
        self.rng.setstate(snapshot["rng_state"])
        return list(self.obs)

    def step(self, action):
        for i in range(self.obs_dim):
            a = action[i % self.act_dim]
            n = self.rng.gauss(0.0, self.noise)
            self.obs[i] = (1.0 - self.drift) * self.obs[i] + self.control_gain * a + n
        reward = -sum(x * x for x in self.obs)
        return self.obs, reward, False, False, {}


class InvertedPendulumTask(BaseTaskEnv):
    def __init__(self):
        super().__init__(obs_dim=4, act_dim=1, drift=0.15, control_gain=0.35, noise=0.005)


class AntTask(BaseTaskEnv):
    def __init__(self):
        super().__init__(obs_dim=8, act_dim=4, drift=0.10, control_gain=0.22, noise=0.01)


class HumanoidTask(BaseTaskEnv):
    def __init__(self):
        super().__init__(obs_dim=10, act_dim=6, drift=0.14, control_gain=0.28, noise=0.008)


TASK_FACTORIES = {
    "inverted_pendulum": InvertedPendulumTask,
    "ant": AntTask,
    "humanoid": HumanoidTask,
}


def make_task_env(name: str):
    if name not in TASK_FACTORIES:
        raise KeyError(f"Unknown task: {name}")
    return TASK_FACTORIES[name]()


def state_norm(state: list[float]) -> float:
    return math.sqrt(sum(x * x for x in state) / max(len(state), 1))
