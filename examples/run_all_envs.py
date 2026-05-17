from __future__ import annotations

from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend
from mpc_warp.core.costs import quadratic_cost
from mpc_warp.envs.registry import ENV_REGISTRY
from mpc_warp.envs.task_envs import make_task_env, state_norm
from mpc_warp.solvers.mppi import MPPIConfig, MPPISolver


def run_task(name: str, max_steps: int, solved_threshold: float) -> tuple[float, float]:
    env = make_task_env(name)
    backend = MujocoWarpBackend(lambda: env)
    x = backend.reset(seed=0)
    initial = state_norm(x)
    action_dim = env.act_dim
    solver = MPPISolver(
        backend,
        MPPIConfig(horizon=12, num_samples=64, noise_sigma=0.3, temperature=0.8, action_dim=action_dim),
        goal=[0.0] * len(x),
        cost_fn=quadratic_cost,
    )
    solver.reset_seed(0)

    for _ in range(max_steps):
        u = solver.command(x)
        step = backend.step(u)
        x = step.state

    final = state_norm(x)
    solved = final <= solved_threshold
    print(f"{name}: initial={initial:.4f} final={final:.4f} solved={solved}")
    if not solved:
        raise RuntimeError(f"Task {name} not solved: final {final:.4f} > threshold {solved_threshold:.4f}")
    return initial, final


def main() -> int:
    for spec in ENV_REGISTRY:
        run_task(spec.name, spec.max_steps, spec.solved_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
