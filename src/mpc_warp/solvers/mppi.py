from __future__ import annotations

import math
import random
from dataclasses import dataclass

from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend
from mpc_warp.core.rollout import rollout_cost


@dataclass
class MPPIConfig:
    horizon: int = 20
    num_samples: int = 64
    noise_sigma: float = 0.3
    temperature: float = 1.0
    action_dim: int = 1


class MPPISolver:
    def __init__(self, backend: MujocoWarpBackend, cfg: MPPIConfig, goal: list[float], cost_fn=None):
        self.backend = backend
        self.cfg = cfg
        self.goal = [float(x) for x in goal]
        self.cost_fn = cost_fn
        self.rng = random.Random(0)
        self.u_nominal = [[0.0 for _ in range(cfg.action_dim)] for _ in range(cfg.horizon)]

    def reset_seed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def _sample_noise(self):
        return [
            [self.rng.gauss(0.0, self.cfg.noise_sigma) for _ in range(self.cfg.action_dim)] for _ in range(self.cfg.horizon)
        ]

    def command(self, x0: list[float]) -> list[float]:
        # Use current live simulator state for all sampled rollouts.
        _ = x0
        snapshot = self.backend.get_state()
        noises = [self._sample_noise() for _ in range(self.cfg.num_samples)]
        costs = []
        for i in range(self.cfg.num_samples):
            trial = [[a + n for a, n in zip(u, nu)] for u, nu in zip(self.u_nominal, noises[i])]
            sim = self.backend.fork_with_state(snapshot)
            costs.append(rollout_cost(sim, trial, self.goal, self.cost_fn))

        beta = min(costs)
        temp = max(self.cfg.temperature, 1e-8)
        ws = [math.exp(-(c - beta) / temp) for c in costs]
        z = sum(ws) if sum(ws) > 0 else 1.0
        ws = [w / z for w in ws]

        for t in range(self.cfg.horizon):
            for j in range(self.cfg.action_dim):
                self.u_nominal[t][j] += sum(ws[i] * noises[i][t][j] for i in range(self.cfg.num_samples))

        u0 = list(self.u_nominal[0])
        self.u_nominal = self.u_nominal[1:] + [[0.0 for _ in range(self.cfg.action_dim)]]
        return u0
