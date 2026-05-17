from mpc_warp.envs.registry import ENV_REGISTRY
from mpc_warp.envs.task_envs import make_task_env, state_norm
from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend
from mpc_warp.solvers.mppi import MPPIConfig, MPPISolver
from mpc_warp.core.costs import quadratic_cost


def test_registry_tasks_are_solved_by_mpc():
    for spec in ENV_REGISTRY:
        env = make_task_env(spec.name)
        backend = MujocoWarpBackend(lambda: env)
        x = backend.reset(seed=0)
        init_norm = state_norm(x)
        solver = MPPISolver(
            backend,
            MPPIConfig(horizon=12, num_samples=64, noise_sigma=0.3, temperature=0.8, action_dim=env.act_dim),
            goal=[0.0] * len(x),
            cost_fn=quadratic_cost,
        )
        solver.reset_seed(0)
        for _ in range(spec.max_steps):
            x = backend.step(solver.command(x)).state

        final_norm = state_norm(x)
        assert final_norm < init_norm
        assert final_norm <= spec.solved_threshold
