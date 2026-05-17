from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend
from mpc_warp.core.costs import quadratic_cost
from mpc_warp.envs.task_envs import make_task_env, state_norm
from mpc_warp.solvers.mppi import MPPIConfig, MPPISolver


@dataclass(frozen=True)
class ParityResult:
    name: str
    initial_norm: float
    final_norm: float
    target_norm: float
    solved: bool


def load_hydrax_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"HydraX manifest not found at {p}. Provide a local exported manifest from hydrax models/tests."
        )
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError("HydraX manifest must be a list of task definitions")
    required = {"name", "max_steps", "solved_threshold", "action_dim"}
    for i, item in enumerate(data):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Task index {i} missing keys: {sorted(missing)}")
    return data


def run_task_with_mpc(name: str, max_steps: int, solved_threshold: float, action_dim: int) -> ParityResult:
    env = make_task_env(name)
    backend = MujocoWarpBackend(lambda: env)
    x = backend.reset(seed=0)
    init = state_norm(x)

    solver = MPPISolver(
        backend,
        MPPIConfig(horizon=12, num_samples=64, noise_sigma=0.3, temperature=0.8, action_dim=action_dim),
        goal=[0.0] * len(x),
        cost_fn=quadratic_cost,
    )
    solver.reset_seed(0)
    for _ in range(max_steps):
        x = backend.step(solver.command(x)).state

    final = state_norm(x)
    return ParityResult(name, init, final, solved_threshold, final <= solved_threshold)


def run_manifest_parity(manifest_path: str | Path) -> list[ParityResult]:
    specs = load_hydrax_manifest(manifest_path)
    results = []
    for spec in specs:
        results.append(
            run_task_with_mpc(
                name=str(spec["name"]),
                max_steps=int(spec["max_steps"]),
                solved_threshold=float(spec["solved_threshold"]),
                action_dim=int(spec["action_dim"]),
            )
        )
    return results
