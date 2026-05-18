from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend
from mpc_warp.core.costs import quadratic_cost
from mpc_warp.solvers.mppi import MPPIConfig, MPPISolver


class TinyEnv:
    def __init__(self):
        self.obs = [0.0, 0.0]

    def reset(self, seed=None):
        self.obs = [0.2, -0.1]
        return self.obs, {}

    def get_internal_state(self):
        return {"obs": list(self.obs)}

    def set_internal_state(self, snapshot):
        self.obs = list(snapshot["obs"])
        return list(self.obs)

    def step(self, action):
        self.obs[0] += 0.1 * action[0]
        self.obs[1] -= 0.1 * action[0]
        return self.obs, -1.0, False, False, {}


def test_mppi_smoke_and_no_planning_side_effects():
    backend = MujocoWarpBackend(TinyEnv)
    x0 = backend.reset(seed=0)
    before = backend.get_state()
    solver = MPPISolver(backend, MPPIConfig(horizon=5, num_samples=16, action_dim=1), goal=[0.0, 0.0], cost_fn=quadratic_cost)
    u = solver.command(x0)
    after = backend.get_state()
    assert len(u) == 1
    assert isinstance(u[0], float)
    assert before == after
