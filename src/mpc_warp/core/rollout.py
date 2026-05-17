from __future__ import annotations

from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend


def rollout_cost(backend: MujocoWarpBackend, x0: list[float], actions: list[list[float]], goal: list[float], cost_fn) -> float:
    total = 0.0
    _ = x0
    for u in actions:
        step = backend.step(u)
        total += cost_fn(step.state, u, goal)
        if step.done:
            break
    return float(total)
