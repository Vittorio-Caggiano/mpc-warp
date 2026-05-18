from __future__ import annotations

from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend


def rollout_cost(backend: MujocoWarpBackend, actions: list[list[float]], goal: list[float], cost_fn=None) -> float:
    total = 0.0
    for u in actions:
        step = backend.step(u)
        # cost_fn is used for synthetic envs (quadratic_cost etc.).
        # For MujocoTaskEnv the cost is already in step.reward (-running_cost),
        # so we use that when no cost_fn is provided, avoiding stale-closure bugs.
        if cost_fn is not None:
            total += cost_fn(step.state, u, goal)
        else:
            total -= step.reward
        if step.done:
            break
    return float(total)
